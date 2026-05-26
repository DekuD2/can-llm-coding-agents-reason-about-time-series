import logging

from pydantic import BaseModel, Field

from factgenie.annotations import AnnotationModelFactory
from factgenie.prompting import transforms as t
from factgenie.prompting.experimental_transforms import CustomToolLoop, McpCodingLoop, ParseRanges, PassData, CallSpec, InterpretCode, Edit, CodingLoop
from factgenie.prompting.strategies import SequentialStrategy, register_llm_eval, register_llm_gen
from factgenie.prompting.text_processing import get_template_sections

logger = logging.getLogger("factgenie")


@register_llm_eval(name="qa_raw_forced")
class QARawForcedStrategy(SequentialStrategy):
    def is_question_answering(self) -> bool:
        return True

    def get_transform_sequence(self) -> list[t.Transform]:
        TEXT = SequentialStrategy.TEXT
        PROMPT = "prompt"
        OPTION_OUTPUT = "option_unparsed"
        OPTIONS = SequentialStrategy.OPTIONS
        REASONING_CONTENT = "reasoning_content"
        CONVERSATION = "conversation"

        system_msg = self.config.get("system_msg", None)
        starts_with = self.config.get("start_with", None)

        assert starts_with is None, "Expected `start_with` to be None."

        return [
            # 1. Ask prompt.
            t.ApplyTemplate(self.config["prompt_template"], PROMPT),
            t.ConverseLLM(
                PROMPT, 
                CONVERSATION,
                system_msg=system_msg,
                start_with=starts_with,
                extractors=t.ConverseLLM.EXTRACTORS_LOG_THINKING,
                choices_path=["data", "options"],
                ensure_completion=True,
            ),
            t.ConversationExtractResponse(CONVERSATION, OPTION_OUTPUT),
            t.ResetIfEmpty(OPTION_OUTPUT),
            t.StringifyConversation(CONVERSATION, CONVERSATION),
            t.ParseOptions(
                OPTION_OUTPUT,
                OPTIONS,
                label="answer",
                choices_path=["data", "options"],
            ),
            t.Metadata(fields=[CONVERSATION]),
        ]


@register_llm_eval(name="qa_code_tool_forced")
class QACodeToolForcedStrategy(SequentialStrategy):
    def is_question_answering(self) -> bool:
        return True

    def get_transform_sequence(self) -> list[t.Transform]:
        TEXT = SequentialStrategy.TEXT
        DATA = SequentialStrategy.DATA
        OPTIONS = SequentialStrategy.OPTIONS
        QUESTION = "question"
        CONVERSATION = "code_loop_conversation"

        OPTION_RESPONSE = "option_response"

        template = self.config["prompt_template"]
        system_msg = self.config.get("system_msg", None)
        extra_args = self.config.get("extra_args", {})
        tool_name = extra_args.get("tool_name", "code")
        no_tool_id = extra_args.get("no_tool_id", False)
        old_description = extra_args.get("old_description", False)
        logger.info(f"Tool name = {tool_name}.")
        logger.info(f"No tool id = {no_tool_id}.")
        logger.info(f"Old description = {old_description}.")

        call_spec = CallSpec(
            "main",
            [
                PassData(
                    "dict_of_dfs",
                    ["data", "json"],
                    """
import json
import pandas as pd
with open({{PATH}}, "r") as f:
    {{NAME}} = json.load(f)
{{NAME}} = {k: pd.DataFrame(v) for k, v in {{NAME}}.items()}
                    """,
                )
            ],
        )

        mcp_loop = McpCodingLoop(
            CONVERSATION,
            interpret_code=InterpretCode(
                McpCodingLoop.FIELD_CODE_INPUT,
                McpCodingLoop.FIELD_CODE_OUTPUT,
                call_spec,
                log_code=True
            ),
            tool_reply=t.ConverseLLM(
                McpCodingLoop.FIELD_CODE_OUTPUT,
                CONVERSATION,
                is_tool_reply=True,
                extractors=t.ConverseLLM.EXTRACTORS_LOG_THINKING,
                is_custom_tool=no_tool_id,
                choices_path=["data", "options"],
                ensure_completion=True,
            ),
            function_name=tool_name,
            old_description=old_description,
        )

        ask_question = (
            t.ApplyTemplate(template, QUESTION),
            t.ConverseLLM(
                QUESTION,
                CONVERSATION,
                system_msg=system_msg,
                completion_kwargs=mcp_loop.completion_kwargs,
                extractors=t.ConverseLLM.EXTRACTORS_LOG_THINKING,
                choices_path=["data", "options"],
                ensure_completion=True,
            ),
        )

        select_option = (
            # This template could need an access to the coding outputs..
            t.ConversationExtractResponse(CONVERSATION, OPTION_RESPONSE),

            # Check the option is non-empty...
            t.ResetIfEmpty(OPTION_RESPONSE),

            # Log the conversation.
            t.StringifyConversation(CONVERSATION, CONVERSATION),
            t.Log("Conversation... \n", field=CONVERSATION, join_by=t.join_string_long),

            # Parse the option.
            t.ParseOptions(
                OPTION_RESPONSE,
                OPTIONS,
                label="answer",
                choices_path=["data", "options"],
            ),
        )

        return [
            *ask_question,
            mcp_loop,
            *select_option,
            t.Metadata(fields=[CONVERSATION]),
        ]
