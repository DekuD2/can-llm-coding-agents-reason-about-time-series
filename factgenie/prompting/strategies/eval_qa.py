import logging

from pydantic import BaseModel, Field

from factgenie.annotations import AnnotationModelFactory
from factgenie.prompting import transforms as t
from factgenie.prompting.experimental_transforms import (
    CallSpec,
    CodingLoop,
    CustomToolLoop,
    Edit,
    InterpretCode,
    McpCodingLoop,
    ParseRanges,
    PassData,
)
from factgenie.prompting.strategies import (
    SequentialStrategy,
    register_llm_eval,
    register_llm_gen,
)
from factgenie.prompting.text_processing import get_template_sections

logger = logging.getLogger("factgenie")


@register_llm_eval(name="qa_raw")
class RawQuestionAnsweringStrategy(SequentialStrategy):
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
            ),
            t.ConversationExtractResponse(CONVERSATION, OPTION_OUTPUT),
            # t.AskPrompt(
            #     PROMPT,
            #     OPTION_OUTPUT,
            #     system_msg,
            #     starts_with,
            #     reasoning_field=REASONING_CONTENT,
            # ),
            t.StringifyConversation(CONVERSATION, CONVERSATION),
            t.ParseOptions(
                OPTION_OUTPUT,
                OPTIONS,
                label="answer",
                choices_path=["data", "options"],
            ),
            t.Metadata(fields=[CONVERSATION]),
        ]


@register_llm_eval(name="qa_coder_agent")
class QuestionAnsweringCoderAgentStrategy(SequentialStrategy):
    def is_question_answering(self) -> bool:
        return True

    def get_transform_sequence(self) -> list[t.Transform]:
        TEXT = SequentialStrategy.TEXT
        DATA = SequentialStrategy.DATA
        OPTIONS = SequentialStrategy.OPTIONS

        DECIDE_PROMPT = "decide_prompt"

        CODE_PROMPT = "code_prompt"
        CONVERSATION = "code_loop_conversation"

        CODE_START = "code_start"
        SELECT_OPTION_PROMPT = "select_option_prompt"
        OPTION_RESPONSE = "option_response"

        super_template = self.config["prompt_template"]
        section_decide = "decide action"
        section_write_code = "write code"
        section_code_start = "code start"
        section_select_option = "select option"

        sections = get_template_sections(
            super_template, [section_decide, section_write_code, section_code_start, section_select_option]
        )

        code_start_template = sections[section_code_start]
        template_code = sections[section_write_code]
        template_select_option = sections[section_select_option]
        template_decide = sections[section_decide]

        ## super_template example:
        # --- decide action ---
        # You are given the following question:
        # (...)
        # --- write code ---
        # You are given the following question:
        # (...)
        # --- code start ---
        # ```python
        # (...)
        # --- select option ---
        # Given current information, (...)

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

        prepare_sentences_and_prompts = (
            # Turn templates into the texts we need.
            t.ApplyTemplate(code_start_template, CODE_START),
            t.ApplyTemplate(template_code, CODE_PROMPT),
            t.ApplyTemplate(template_decide, DECIDE_PROMPT),
        )

        CODE_START_IF_NONEMPTY = CODE_START if len(code_start_template) > 0 else None
        assert CODE_START_IF_NONEMPTY is None, "Expected `start_with` to be None."
        if CODE_START_IF_NONEMPTY is None:
            logger.info("Coding loop does not have a CODE_START")
        else:
            logger.info("Coding has a CODE_START")

        coding_loop = CodingLoop(
            decision_prompt=t.ConverseLLM(DECIDE_PROMPT, CodingLoop.FIELD_CONVERSATION),
            code_prompt=t.ConverseLLM(
                CODE_PROMPT, CodingLoop.FIELD_CONVERSATION, start_with_field=CODE_START_IF_NONEMPTY
            ),
            extract_code=t.ExtractCodeBlock(
                CodingLoop.FIELD_CODE_UNEXTRACTED,
                CodingLoop.FIELD_CODE_TO_EXECUTE,
                language="python",
                join_occurances=True,
                remove_from_input=False,
            ),
            interpret_code=InterpretCode(
                CodingLoop.FIELD_CODE_TO_EXECUTE, CodingLoop.FIELD_CODE_OUTPUT, call_spec, log_code=True
            ),
            output_field=CONVERSATION,
            max_iters=10,
        )

        select_option = (
            # This template could need an access to the coding outputs..
            t.ApplyTemplate(template_select_option, SELECT_OPTION_PROMPT),
            t.ConverseLLM(SELECT_OPTION_PROMPT, CONVERSATION),
            t.ConversationExtractResponse(CONVERSATION, OPTION_RESPONSE),
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
            *prepare_sentences_and_prompts,
            coding_loop,
            *select_option,
            t.Metadata(fields=[CONVERSATION]),
        ]


@register_llm_eval(name="qa_hybrid_agent")
class QuestionAnsweringHybridAgentStrategy(SequentialStrategy):
    def is_question_answering(self) -> bool:
        return True

    def get_transform_sequence(self) -> list[t.Transform]:
        TEXT = SequentialStrategy.TEXT
        DATA = SequentialStrategy.DATA
        OPTIONS = SequentialStrategy.OPTIONS

        DECIDE_PROMPT = "decide_prompt"

        CODE_PROMPT = "code_prompt"
        CONVERSATION = "code_loop_conversation"

        CODE_START = "code_start"
        SELECT_OPTION_PROMPT = "select_option_prompt"
        OPTION_RESPONSE = "option_response"
        INITIAL_INFO = "initial"

        super_template = self.config["prompt_template"]
        section_initial = "initial"
        section_decide = "decide action"
        section_write_code = "write code"
        section_code_start = "code start"
        section_select_option = "select option"

        thought_context = self.extra_args.get("thought_context", False)
        logger.info(f"Thought context = {thought_context}")

        sections = get_template_sections(
            super_template,
            [section_initial, section_decide, section_write_code, section_code_start, section_select_option],
        )

        code_start_template = sections[section_code_start]
        template_initial = sections[section_initial]
        template_code = sections[section_write_code]
        template_select_option = sections[section_select_option]
        template_decide = sections[section_decide]

        ## super_template example:
        # --- decide action ---
        # You are given the following question:
        # (...)
        # --- write code ---
        # You are given the following question:
        # (...)
        # --- code start ---
        # ```python
        # (...)
        # --- select option ---
        # Given current information, (...)

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

        prepare_sentences_and_prompts = (
            # Turn templates into the texts we need.
            t.ApplyTemplate(code_start_template, CODE_START),
            t.ApplyTemplate(template_code, CODE_PROMPT),
            t.ApplyTemplate(template_decide, DECIDE_PROMPT),
            # An initial prompt (containing the time series raw).
            t.ApplyTemplate(template_initial, INITIAL_INFO),
            t.ConversationAppendResponse(CodingLoop.FIELD_CONVERSATION, INITIAL_INFO, role=t.ConverseLLM.USER),
        )

        CODE_START_IF_NONEMPTY = CODE_START if len(code_start_template) > 0 else None
        assert CODE_START_IF_NONEMPTY is None, "Expected `start_with` to be None."
        if CODE_START_IF_NONEMPTY is None:
            logger.info("Coding loop does not have a CODE_START")
        else:
            logger.info("Coding has a CODE_START")

        coding_loop = CodingLoop(
            decision_prompt=t.ConverseLLM(
                DECIDE_PROMPT, CodingLoop.FIELD_CONVERSATION, thought_context=thought_context
            ),
            code_prompt=t.ConverseLLM(
                CODE_PROMPT,
                CodingLoop.FIELD_CONVERSATION,
                start_with_field=CODE_START_IF_NONEMPTY,
                thought_context=thought_context,
            ),
            extract_code=t.ExtractCodeBlock(
                CodingLoop.FIELD_CODE_UNEXTRACTED,
                CodingLoop.FIELD_CODE_TO_EXECUTE,
                language="python",
                join_occurances=True,
                remove_from_input=False,
            ),
            interpret_code=InterpretCode(
                CodingLoop.FIELD_CODE_TO_EXECUTE, CodingLoop.FIELD_CODE_OUTPUT, call_spec, log_code=True
            ),
            output_field=CONVERSATION,
            max_iters=10,
        )

        select_option = (
            # This template could need an access to the coding outputs..
            t.ApplyTemplate(template_select_option, SELECT_OPTION_PROMPT),
            t.ConverseLLM(SELECT_OPTION_PROMPT, CONVERSATION, thought_context=thought_context),
            t.ConversationExtractResponse(CONVERSATION, OPTION_RESPONSE),
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
            *prepare_sentences_and_prompts,
            coding_loop,
            *select_option,
            t.Metadata(fields=[CONVERSATION]),
        ]


@register_llm_eval(name="qa2_hybrid_agent")
class QuestionAnsweringHybridAgentStrategy2(SequentialStrategy):
    def is_question_answering(self) -> bool:
        return True

    def get_transform_sequence(self) -> list[t.Transform]:
        TEXT = SequentialStrategy.TEXT
        DATA = SequentialStrategy.DATA
        OPTIONS = SequentialStrategy.OPTIONS

        DECIDE_PROMPT = "decide_prompt"

        CODE_PROMPT = "code_prompt"
        CONVERSATION = "code_loop_conversation"

        CODE_START = "code_start"
        SELECT_OPTION_PROMPT = "select_option_prompt"
        OPTION_RESPONSE = "option_response"
        INITIAL_INFO = "initial"

        super_template = self.config["prompt_template"]
        section_initial = "initial"
        section_decide = "decide action"
        section_write_code = "write code"
        section_code_start = "code start"
        section_select_option = "select option"

        thought_context = self.extra_args.get("thought_context", False)
        logger.info(f"Thought context = {thought_context}")

        sections = get_template_sections(
            super_template,
            [section_initial, section_decide, section_write_code, section_code_start, section_select_option],
        )

        code_start_template = sections[section_code_start]
        template_initial = sections[section_initial]
        template_code = sections[section_write_code]
        template_select_option = sections[section_select_option]
        template_decide = sections[section_decide]

        ## super_template example:
        # --- decide action ---
        # You are given the following question:
        # (...)
        # --- write code ---
        # You are given the following question:
        # (...)
        # --- code start ---
        # ```python
        # (...)
        # --- select option ---
        # Given current information, (...)

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

        prepare_sentences_and_prompts = (
            # Turn templates into the texts we need.
            t.ApplyTemplate(code_start_template, CODE_START),
            t.ApplyTemplate(template_code, CODE_PROMPT),
            t.ApplyTemplate(template_decide, DECIDE_PROMPT),
            # An initial prompt (containing the time series raw).
            t.ApplyTemplate(template_initial, INITIAL_INFO),
            t.ConversationAppendResponse(CodingLoop.FIELD_CONVERSATION, INITIAL_INFO, role=t.ConverseLLM.USER),
        )

        CODE_START_IF_NONEMPTY = CODE_START if len(code_start_template) > 0 else None
        assert CODE_START_IF_NONEMPTY is None, "Expected `start_with` to be None."
        if CODE_START_IF_NONEMPTY is None:
            logger.info("Coding loop does not have a CODE_START")
        else:
            logger.info("Coding has a CODE_START")

        coding_loop = CodingLoop(
            decision_prompt=t.ConverseLLM(
                DECIDE_PROMPT, CodingLoop.FIELD_CONVERSATION, thought_context=thought_context
            ),
            code_prompt=t.ConverseLLM(
                CODE_PROMPT,
                CodingLoop.FIELD_CONVERSATION,
                start_with_field=CODE_START_IF_NONEMPTY,
                thought_context=thought_context,
            ),
            extract_code=t.ExtractCodeBlock(
                CodingLoop.FIELD_CODE_UNEXTRACTED,
                CodingLoop.FIELD_CODE_TO_EXECUTE,
                language="python",
                join_occurances=True,
                remove_from_input=False,
            ),
            interpret_code=InterpretCode(
                CodingLoop.FIELD_CODE_TO_EXECUTE, CodingLoop.FIELD_CODE_OUTPUT, call_spec, log_code=True
            ),
            output_field=CONVERSATION,
            max_iters=10,
        )

        select_option = (
            # This template could need an access to the coding outputs..
            t.ApplyTemplate(template_select_option, SELECT_OPTION_PROMPT),
            t.ConverseLLM(SELECT_OPTION_PROMPT, CONVERSATION, thought_context=thought_context),
            t.ConversationExtractResponse(CONVERSATION, OPTION_RESPONSE),
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
            *prepare_sentences_and_prompts,
            coding_loop,
            *select_option,
            t.Metadata(fields=[CONVERSATION]),
        ]


@register_llm_gen(name="tsa_hybrid_agent")
class AnomnalyDetectionHybridAgentStrategy(SequentialStrategy):
    def get_transform_sequence(self) -> list[t.Transform]:
        TEXT = SequentialStrategy.TEXT
        DATA = SequentialStrategy.DATA
        ANOMALIES = SequentialStrategy.ANOMALIES
        OUTPUT = SequentialStrategy.OUTPUT

        DECIDE_PROMPT = "decide_prompt"

        CODE_PROMPT = "code_prompt"
        CONVERSATION = "code_loop_conversation"

        CODE_START = "code_start"
        SELECT_OPTION_PROMPT = "select_option_prompt"
        RANGES_RESPONSE = "ranges_response"
        INITIAL_INFO = "initial"

        super_template = self.config["prompt_template"]
        section_initial = "initial"
        section_decide = "decide action"
        section_write_code = "write code"
        section_code_start = "code start"
        section_select_option = "select option"

        thought_context = self.extra_args.get("thought_context", False)
        logger.info(f"Thought context = {thought_context}")

        sections = get_template_sections(
            super_template,
            [section_initial, section_decide, section_write_code, section_code_start, section_select_option],
        )

        code_start_template = sections[section_code_start]
        template_initial = sections[section_initial]
        template_code = sections[section_write_code]
        template_select_option = sections[section_select_option]
        template_decide = sections[section_decide]

        ## super_template example:
        # --- decide action ---
        # You are given the following question:
        # (...)
        # --- write code ---
        # You are given the following question:
        # (...)
        # --- code start ---
        # ```python
        # (...)
        # --- select option ---
        # Given current information, (...)

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
                    """,
                )
            ],
        )

        prepare_sentences_and_prompts = (
            # Turn templates into the texts we need.
            t.ApplyTemplate(code_start_template, CODE_START),
            t.ApplyTemplate(template_code, CODE_PROMPT),
            t.ApplyTemplate(template_decide, DECIDE_PROMPT),
            # An initial prompt (containing the time series raw).
            t.ApplyTemplate(template_initial, INITIAL_INFO),
            t.ConversationAppendResponse(CodingLoop.FIELD_CONVERSATION, INITIAL_INFO, role=t.ConverseLLM.USER),
        )

        CODE_START_IF_NONEMPTY = CODE_START if len(code_start_template) > 0 else None
        assert CODE_START_IF_NONEMPTY is None, "Expected `start_with` to be None."
        if CODE_START_IF_NONEMPTY is None:
            logger.info("Coding loop does not have a CODE_START")
        else:
            logger.info("Coding has a CODE_START")

        coding_loop = CodingLoop(
            decision_prompt=t.ConverseLLM(
                DECIDE_PROMPT, CodingLoop.FIELD_CONVERSATION, thought_context=thought_context
            ),
            code_prompt=t.ConverseLLM(
                CODE_PROMPT,
                CodingLoop.FIELD_CONVERSATION,
                start_with_field=CODE_START_IF_NONEMPTY,
                thought_context=thought_context,
            ),
            extract_code=t.ExtractCodeBlock(
                CodingLoop.FIELD_CODE_UNEXTRACTED,
                CodingLoop.FIELD_CODE_TO_EXECUTE,
                language="python",
                join_occurances=True,
                remove_from_input=False,
            ),
            interpret_code=InterpretCode(
                CodingLoop.FIELD_CODE_TO_EXECUTE, CodingLoop.FIELD_CODE_OUTPUT, call_spec, log_code=True
            ),
            output_field=CONVERSATION,
            max_iters=10,
        )

        select_option = (
            # This template could need an access to the coding outputs..
            t.ApplyTemplate(template_select_option, SELECT_OPTION_PROMPT),
            t.ConverseLLM(SELECT_OPTION_PROMPT, CONVERSATION, thought_context=thought_context),
            t.ConversationExtractResponse(CONVERSATION, RANGES_RESPONSE),
            # Log the conversation.
            t.StringifyConversation(CONVERSATION, CONVERSATION),
            t.Log("Conversation... \n", field=CONVERSATION, join_by=t.join_string_long),
            # Parse the anomalies.
            ParseRanges(RANGES_RESPONSE, ANOMALIES),
        )

        return [
            *prepare_sentences_and_prompts,
            coding_loop,
            *select_option,
            t.Duplicate(RANGES_RESPONSE, OUTPUT),  # To not raise an error and also to have something to show.
            t.Metadata(fields=[CONVERSATION]),
        ]


@register_llm_gen(name="tsa_raw")
class RawAnomalyStrategy(SequentialStrategy):
    def get_transform_sequence(self) -> list[t.Transform]:
        TEXT = SequentialStrategy.TEXT
        PROMPT = "prompt"
        RANGES_OUTPUT = "ranges_unparsed"
        OUTPUT = SequentialStrategy.OUTPUT
        ANOMALIES = SequentialStrategy.ANOMALIES
        REASONING_CONTENT = "reasoning_content"
        CONVERSATION = "conversation"

        system_msg = self.config.get("system_msg", None)
        starts_with = self.config.get("start_with", None)
        if starts_with == "":
            starts_with = None

        assert starts_with is None, "Expected `start_with` to be None."

        return [
            # 1. Ask prompt.
            t.ApplyTemplate(self.config["prompt_template"], PROMPT),
            t.ConverseLLM(
                PROMPT,
                CONVERSATION,
                system_msg=system_msg,
                start_with=starts_with,
            ),
            t.ConversationExtractResponse(CONVERSATION, RANGES_OUTPUT),
            # t.AskPrompt(
            #     PROMPT,
            #     OPTION_OUTPUT,
            #     system_msg,
            #     starts_with,
            #     reasoning_field=REASONING_CONTENT,
            # ),
            ParseRanges(RANGES_OUTPUT, ANOMALIES),
            t.Duplicate(RANGES_OUTPUT, OUTPUT),  # To not raise an error and also to have something to show.
            t.Metadata(fields=[CONVERSATION]),
        ]


# --------------------------------------------- #
# -------------------- MCP -------------------- #
# --------------------------------------------- #
# (actually just tool using)


@register_llm_eval(name="qa_mcp_agent")
class QuestionAnsweringMcpAgentStrategy(SequentialStrategy):
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
                McpCodingLoop.FIELD_CODE_INPUT, McpCodingLoop.FIELD_CODE_OUTPUT, call_spec, log_code=True
            ),
            tool_reply=t.ConverseLLM(
                McpCodingLoop.FIELD_CODE_OUTPUT,
                CONVERSATION,
                is_tool_reply=True,
                extractors=t.ConverseLLM.EXTRACTORS_LOG_THINKING,
                is_custom_tool=no_tool_id,
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
            ),
        )

        select_option = (
            # This template could need an access to the coding outputs..
            t.ConversationExtractResponse(CONVERSATION, OPTION_RESPONSE),
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


@register_llm_eval(name="qa_mcp_interactive")
class QuestionAnsweringMcpAgentInteractiveStrategy(SequentialStrategy):
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
globals().update({{NAME}})
                    """,
                )
            ],
        )

        mcp_loop = McpCodingLoop(
            CONVERSATION,
            interpret_code=InterpretCode(
                McpCodingLoop.FIELD_CODE_INPUT, McpCodingLoop.FIELD_CODE_OUTPUT, call_spec, log_code=True
            ),
            tool_reply=t.ConverseLLM(
                McpCodingLoop.FIELD_CODE_OUTPUT,
                CONVERSATION,
                is_tool_reply=True,
                extractors=t.ConverseLLM.EXTRACTORS_LOG_THINKING,
            ),
            interactive=True,
            max_iters=14,
        )

        ask_question = (
            t.ApplyTemplate(template, QUESTION),
            t.ConverseLLM(
                QUESTION,
                CONVERSATION,
                system_msg=system_msg,
                completion_kwargs=mcp_loop.completion_kwargs,
                extractors=t.ConverseLLM.EXTRACTORS_LOG_THINKING,
            ),
        )

        select_option = (
            # This template could need an access to the coding outputs..
            t.ConversationExtractResponse(CONVERSATION, OPTION_RESPONSE),
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


@register_llm_eval(name="qa_custom_interactive")
class QuestionAnsweringMcpAgentInteractiveCustomStrategy(SequentialStrategy):
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
globals().update({{NAME}})
                    """,
                )
            ],
        )

        tool_loop = CustomToolLoop(
            CONVERSATION,
            interpret_code=InterpretCode(
                McpCodingLoop.FIELD_CODE_INPUT, McpCodingLoop.FIELD_CODE_OUTPUT, call_spec, log_code=True
            ),
            tool_reply=t.ConverseLLM(
                McpCodingLoop.FIELD_CODE_OUTPUT,
                CONVERSATION,
                is_tool_reply=True,
                is_custom_tool=True,
                extractors=t.ConverseLLM.EXTRACTORS_LOG_THINKING,
            ),
            interactive=True,
            max_iters=14,
        )

        ask_question = (
            t.ApplyTemplate(template, QUESTION),
            t.ConverseLLM(
                QUESTION, CONVERSATION, system_msg=system_msg, extractors=t.ConverseLLM.EXTRACTORS_LOG_THINKING
            ),
        )

        select_option = (
            # This template could need an access to the coding outputs..
            t.ConversationExtractResponse(CONVERSATION, OPTION_RESPONSE),
            # t.ExtractRegex(OPTION_RESPONSE, None, remove_from_input=True, flags=0),
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
            tool_loop,
            *select_option,
            t.Metadata(fields=[CONVERSATION]),
        ]
