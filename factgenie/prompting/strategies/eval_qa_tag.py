import logging

from pydantic import BaseModel, Field

from factgenie.annotations import AnnotationModelFactory
from factgenie.prompting import transforms as t
from factgenie.prompting.experimental_transforms import CustomToolLoop, AgenticCodingLoop, ParseRanges, PassData, CallSpec, InterpretCode, Edit, CodingLoop
from factgenie.prompting.strategies import SequentialStrategy, register_llm_eval, register_llm_gen
from factgenie.prompting.text_processing import get_template_sections

logger = logging.getLogger("factgenie")


@register_llm_eval(name="qa_raw_tag")
class QARawTagStrategy(SequentialStrategy):
    def is_question_answering(self) -> bool:
        return True

    def get_transform_sequence(self) -> list[t.Transform]:
        TEXT = SequentialStrategy.TEXT
        PROMPT = "prompt"
        OPTION_OUTPUT = "option_unparsed"
        OPTION_EXTRACTED = "option_extracted"
        OPTIONS = SequentialStrategy.OPTIONS
        REASONING_CONTENT = "reasoning_content"
        CONVERSATION = "conversation"

        system_msg = self.config.get("system_msg", None)
        starts_with = self.config.get("start_with", None)

        extra_args = self.config.get("extra_args", {})
        out_of_time_message = extra_args.get("out_of_time_message", "The time to think is up, output now.")
        model_prepends = extra_args.get("model_prepends", None)
        thought_context = extra_args.get("thought_context", False)
        thought_context_on_completion = extra_args.get("thought_context_on_completion", False)

        assert starts_with is None, "Expected `start_with` to be None."

        # Make sure qwen is used correctly
        model_name = self.config.get("model", "").lower()
        if "qwen" in model_name:
            logger.info("Qwen detected, doing some auto-config:")
            model_prepends = "<think>"
            logger.info(' - model_prepends = "<think>"')
            thought_context = True
            logger.info(' - thought_context = True')
        else:
            logger.info("Qwen not detected, not doing anything")

        return [
            # 1. Ask prompt.
            t.ApplyTemplate(self.config["prompt_template"], PROMPT),
            t.ConverseLLM(
                PROMPT, 
                CONVERSATION,
                system_msg=system_msg,
                start_with=starts_with,
                extractors=t.ConverseLLM.EXTRACTORS_LOG_THINKING,
                ensure_completion=True,
                out_of_time_message=out_of_time_message,
                model_prepends=model_prepends,
                thought_context=thought_context,
                thought_context_on_completion=thought_context_on_completion,
            ),
            t.ConversationExtractResponse(CONVERSATION, OPTION_OUTPUT),

            # Remove <think> tags if present (for olmo)
            t.ExtractTag(OPTION_OUTPUT, None, tag="think", log_as="thinking", remove_from_input=True),

            # Extract the answer from the tag.
            t.Log(text="Model output: ", field=OPTION_OUTPUT),
            t.ExtractTag(OPTION_OUTPUT, OPTION_EXTRACTED, "answer", log_as="ANSWERING"),
            t.ResetIfEmpty(OPTION_EXTRACTED),
            t.ParseOptions(
                OPTION_EXTRACTED,
                OPTIONS,
                label="answer",
                choices_path=["data", "options"],
            ),

            t.StringifyConversation(CONVERSATION, CONVERSATION),
            t.Log("Conversation... \n", field=CONVERSATION, join_by=t.join_string_long),
            t.Metadata(fields=[CONVERSATION]),
        ]


@register_llm_eval(name="qa_code_tool_tag")
class QACodeToolTagStrategy(SequentialStrategy):
    def is_question_answering(self) -> bool:
        return True

    def get_transform_sequence(self) -> list[t.Transform]:
        TEXT = SequentialStrategy.TEXT
        DATA = SequentialStrategy.DATA
        OPTIONS = SequentialStrategy.OPTIONS
        QUESTION = "question"
        CONVERSATION = "code_loop_conversation"

        OPTION_RESPONSE = "option_response"
        OPTION_EXTRACTED = "option_extracted"

        template = self.config["prompt_template"]
        system_msg = self.config.get("system_msg", None)

        extra_args = self.config.get("extra_args", {})
        tool_name = extra_args.get("tool_name", "code")
        no_tool_id = extra_args.get("no_tool_id", False)
        thought_context = extra_args.get("thought_context", False)
        thought_context_on_completion = extra_args.get("thought_context_on_completion", False)
        old_description = extra_args.get("old_description", True)
        out_of_time_message = extra_args.get("out_of_time_message", "The time to think is up, output now.")
        model_prepends = extra_args.get("model_prepends", None)

        # Make sure qwen is used correctly
        model_name = self.config.get("model", "").lower()
        if "qwen" in model_name:
            logger.info("Qwen detected, doing some auto-config:")
            model_prepends = "<think>"
            logger.info(' - model_prepends = "<think>"')
            thought_context = True
            logger.info(' - thought_context = True')
        else:
            logger.info("Qwen not detected, not doing anything")

        tsfu = extra_args.get("tsfu", False)
        logger.info(f"tsfu = {tsfu}.")
        logger.info(f"Tool name = {tool_name}.")
        logger.info(f"No tool id = {no_tool_id}.")
        logger.info(f"Old description = {old_description}.")

        if tsfu:
            call_spec = CallSpec(
                "main",
                [
                    PassData(
                        "df",
                        ["data", "ts"],
                        """
import json
import pandas as pd
with open({{PATH}}, "r") as f:
    {{NAME}} = json.load(f)
{{NAME}} = pd.DataFrame({{NAME}})
{{NAME}}['time'] = pd.to_datetime({{NAME}}['time'])
                        """,
                    )
                ],
            )
        else:
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

        coding_loop = AgenticCodingLoop(
            CONVERSATION,
            interpret_code=InterpretCode(
                AgenticCodingLoop.FIELD_CODE_INPUT,
                AgenticCodingLoop.FIELD_CODE_OUTPUT,
                call_spec,
                log_code=True
            ),
            tool_reply=t.ConverseLLM(
                AgenticCodingLoop.FIELD_CODE_OUTPUT,
                CONVERSATION,
                is_tool_reply=True,
                extractors=t.ConverseLLM.EXTRACTORS_LOG_THINKING,
                is_custom_tool=no_tool_id,
                ensure_completion=True,
                out_of_time_message=out_of_time_message,
                model_prepends=model_prepends,
                thought_context=thought_context,
                thought_context_on_completion=thought_context_on_completion,
            ),
            function_name=tool_name,
            old_description=old_description,
            max_iters=14,
        )

        ask_question = (
            t.ApplyTemplate(template, QUESTION),
            t.ConverseLLM(
                QUESTION,
                CONVERSATION,
                system_msg=system_msg,
                completion_kwargs=coding_loop.completion_kwargs,
                extractors=t.ConverseLLM.EXTRACTORS_LOG_THINKING,
                ensure_completion=True,
                out_of_time_message=out_of_time_message,
                model_prepends=model_prepends,
                thought_context=thought_context,
                thought_context_on_completion=thought_context_on_completion,
            ),
        )

        select_option = (
            # This template could need an access to the coding outputs..
            t.ConversationExtractResponse(CONVERSATION, OPTION_RESPONSE),


            # Log the conversation.
            t.StringifyConversation(CONVERSATION, CONVERSATION),
            t.Log("Conversation... \n", field=CONVERSATION, join_by=t.join_string_long),

            # Remove <think> tags if present (for olmo)
            t.ExtractTag(OPTION_RESPONSE, None, tag="think", log_as="thinking", remove_from_input=True),

            # Extract the answer from the tag.
            t.Log(text="Model output: ", field=OPTION_RESPONSE),
            t.ExtractTag(OPTION_RESPONSE, OPTION_EXTRACTED, "answer", log_as="ANSWERING"),
            t.ResetIfEmpty(OPTION_EXTRACTED),
            t.ParseOptions(
                OPTION_EXTRACTED,
                OPTIONS,
                label="answer",
                choices_path=["data", "options"],
            ),
        )

        return [
            *ask_question,
            coding_loop,
            *select_option,
            t.Metadata(fields=[CONVERSATION]),
        ]
