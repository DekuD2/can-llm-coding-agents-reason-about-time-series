import logging

from sqlalchemy.types import STRINGTYPE

from factgenie.annotations import AnnotationModelFactory
from factgenie.prompting import transforms as t
from factgenie.prompting.strategies import register_llm_gen, SequentialStrategy
from factgenie.prompting.fact_splitting import DebugShowFacts
from factgenie.prompting.experimental_transforms import PassData, CallSpec, InterpretCode, Edit, CodingLoop
from factgenie.prompting.text_processing import get_template_sections

logger = logging.getLogger("factgenie")


@register_llm_gen(name="gen_coder_agent")
class GenCoderAgentStrategy(SequentialStrategy):
    def is_question_answering(self) -> bool:
        return True

    def get_transform_sequence(self) -> list[t.Transform]:
        DATA = SequentialStrategy.DATA
        OUTPUT = SequentialStrategy.OUTPUT

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
        section_write_output = "write output"

        sections = get_template_sections(
            super_template, [section_decide, section_write_code, section_code_start, section_write_output]
        )

        code_start_template = sections[section_code_start]
        template_code = sections[section_write_code]
        template_write_output = sections[section_write_output]
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
        # --- write output ---
        # Given current information, (...)

        call_spec = CallSpec(
            "main",
            [
                PassData(
                    "df",
                    ["data", "single-dim", "pandas"],
                    # TODO: Use this in the other coders as well.
                    """
{{NAME}} = pd.read_json({{PATH}})
{{NAME}}.set_index('time')
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
        )

        coding_loop = CodingLoop(
            decision_prompt=t.ConverseLLM(DECIDE_PROMPT, CodingLoop.FIELD_CONVERSATION),
            code_prompt=t.ConverseLLM(CODE_PROMPT, CodingLoop.FIELD_CONVERSATION, start_with_field=CODE_START),
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

        write_output = (
            # This template could need an access to the coding outputs..
            t.ApplyTemplate(template_write_output, SELECT_OPTION_PROMPT),
            t.ConverseLLM(SELECT_OPTION_PROMPT, CONVERSATION),
            t.ConversationExtractResponse(CONVERSATION, OUTPUT),
        )

        log_and_metadata = (
            # Log the conversation.
            t.StringifyConversation(CONVERSATION, CONVERSATION),
            t.Log("Conversation... \n", field=CONVERSATION, join_by=t.join_string_long),
            t.Metadata(fields=[CONVERSATION]),
        )

        return [*prepare_sentences_and_prompts, coding_loop, *write_output, *log_and_metadata]
