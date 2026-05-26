import logging

from sqlalchemy.types import STRINGTYPE

from factgenie.annotations import AnnotationModelFactory
from factgenie.prompting import transforms as t
from factgenie.prompting.strategies import register_llm_eval, SequentialStrategy
from factgenie.prompting.fact_splitting import DebugShowFacts
from factgenie.prompting.experimental_transforms import PassData, CallSpec, InterpretCode, Edit, CodingLoop
from factgenie.prompting.text_processing import get_template_sections

logger = logging.getLogger("factgenie")


def make_call_function_script(function_name: str, files_args: dict = {}):
    """
    Example:
        Args:
            fucntion_name = "main"
            files_args = {"df", "./data.csv"}

        Returns:
            df = pandas.read_csv("./data.csv")
            main(df)
    """


@register_llm_eval("sentence_split_coder")
class CoderStrategy(SequentialStrategy):
    def get_transform_sequence(self) -> list[t.Transform]:
        TEXT = SequentialStrategy.TEXT
        DATA = SequentialStrategy.DATA
        PART = "part"
        GATHER_PROMPT = "gather_prompt"
        GATHER_RESPONSE = "gather_response"
        CODE = "code"
        START_WITH = "code_start"
        CODE_OUTPUT = "code_result"
        THINKING_TRACE = "thinking_trace"
        ANNOTATION_PROMPT = "annotation_prompt"
        ANNOTATION_TEXT = "annotation_text"
        ANNOTATION_RESPONSE = "annotation_response"
        ANNOTATIONS = SequentialStrategy.ANNOTATIONS
        EXTRACTED = "extracted"

        super_template = self.config["prompt_template"]
        section_initial = "initial"
        section_code_start = "code start"
        section_annotate = "annotate"
        sections = get_template_sections(super_template, [section_initial, section_code_start, section_annotate])
        code_start_template = sections[section_code_start]
        template_gather = sections[section_initial]
        template_annotate = sections[section_annotate]

        ## super_template example:
        # --- initial ---
        # You are given the following annotation:
        # (...)
        # --- code start ---
        # ```python
        # (...)
        # --- annotate ---
        # Given these results, (...)

        # I could make the Modules return args they need. Possibly they could even be sub-dictionaries.
        annotation_span_categories = self.config["annotation_span_categories"]
        annotation_overlap_allowed = self.config.get("annotation_overlap_allowed", False)
        with_reason = self.extra_args.get("with_reason", True)
        output_validation_model = AnnotationModelFactory.get_output_model(with_reason)

        # ignore_keywords = self.extra_args.get("ignore_keywords", None)
        # # Returns true if the response doesn't match one of the ignore phrases.
        # #
        # def is_relevant(response: str, api: ModelAPI):
        #     if ignore_keywords is None:
        #         return True
        #     else:
        #         # Either of the ignore keywords with up to 7 paddings on either side (e.g. for a ".")
        #         regex = f"^.{0,7}(?:{'|'.join(ignore_keywords)}).{0,7}$"
        #         return not re.match(regex, response, re.DOTALL)

        call_spec = CallSpec(
            "main",
            [
                PassData(
                    "df",
                    ["data", "single-dim", "pandas"],
                    "{{NAME}} = pd.read_json({{PATH}}); {{NAME}}['time'] = pd.to_datetime({{NAME}}['time'])",
                )
            ],
        )

        return [
            # 1. Split sentences
            t.SentenceSplit(TEXT, PART),
            t.LimitEntries(3),  # TODO: This is only for debug!!! Remove later!!!
            # 2. Ask for a code
            t.ApplyTemplate(code_start_template, START_WITH),
            t.ApplyTemplate(template_gather, GATHER_PROMPT),
            t.AskPrompt(GATHER_PROMPT, GATHER_RESPONSE, start_with_field=START_WITH, keep_start_with=True),
            # For sandbox experimenting, use this instead of AskPrompt.
            # t.Put(CODE_PUT, GATHER_RESPONSE),
            # TODO: Add again when thinking
            # t.ExtractTag(GATHER_RESPONSE, output_field=None, tag="think", remove_from_input=True, log_as="THINKING"),
            # t.Filter([GATHER_PROMPT], is_relevant),
            # 3. Intepret code
            t.ExtractCodeBlock(
                GATHER_RESPONSE, CODE, language="python", join_occurances=True, remove_from_input=False
            ),  # , log_as="CODE"),
            InterpretCode(CODE, CODE_OUTPUT, call_spec, log_code=True),
            # 4. Ask to annotate code result
            t.ApplyTemplate(template_annotate, ANNOTATION_PROMPT),
            t.AskPrompt(ANNOTATION_PROMPT, ANNOTATION_RESPONSE),
            t.Log("Annotation response: ", field=ANNOTATION_RESPONSE, join_by=t.join_string_long),
            t.ExtractTag(
                ANNOTATION_RESPONSE,
                THINKING_TRACE,
                tag="think",
                join_occurances=True,
                remove_from_input=True,
                log_as="THINKING",
            ),
            # 5. Join answers
            t.Unify(annotation_fields=[ANNOTATION_RESPONSE], join_strings_by=t.join_string_long),
            t.ExtractJson(ANNOTATION_RESPONSE, EXTRACTED),
            t.ParseAnnotations(
                EXTRACTED, ANNOTATIONS, annotation_span_categories, annotation_overlap_allowed, output_validation_model
            ),
            # Metadata
            t.Metadata(fields=[GATHER_PROMPT, ANNOTATION_PROMPT]),
        ]


@register_llm_eval("sentence_split_coder_agent")
class CoderAgentStrategy(SequentialStrategy):
    def get_transform_sequence(self) -> list[t.Transform]:
        TEXT = SequentialStrategy.TEXT
        DATA = SequentialStrategy.DATA
        PART = "part"

        DECIDE_PROMPT = "decide_prompt"

        CODE_PROMPT = "code_prompt"
        CODE_RESPONSE = "code_response"
        CODE = "code"
        CONVERSATION = "code_loop_conversation"

        START_WITH = "code_start"
        CODE_OUTPUT = "code_result"
        THINKING_TRACE = "thinking_trace"
        ANNOTATION_PROMPT = "annotation_prompt"
        ANNOTATION_TEXT = "annotation_text"
        ANNOTATION_RESPONSE = "annotation_response"
        ANNOTATIONS = SequentialStrategy.ANNOTATIONS
        EXTRACTED = "extracted"

        super_template = self.config["prompt_template"]
        section_decide = "decide action"
        section_write_code = "write code"
        section_code_start = "code start"
        section_annotate = "annotate"

        sections = get_template_sections(
            super_template, [section_decide, section_write_code, section_code_start, section_annotate]
        )

        code_start_template = sections[section_code_start]
        template_code = sections[section_write_code]
        template_annotate = sections[section_annotate]
        template_decide = sections[section_decide]

        ## super_template example:
        # --- decide action ---
        # You are given the following annotation:
        # (...)
        # --- write code ---
        # You are given the following annotation:
        # (...)
        # --- code start ---
        # ```python
        # (...)
        # --- annotate ---
        # Given these results, (...)

        # I could make the Modules return args they need. Possibly they could even be sub-dictionaries.
        annotation_span_categories = self.config["annotation_span_categories"]
        annotation_overlap_allowed = self.config.get("annotation_overlap_allowed", False)
        with_reason = self.extra_args.get("with_reason", True)
        output_validation_model = AnnotationModelFactory.get_output_model(with_reason)

        # ignore_keywords = self.extra_args.get("ignore_keywords", None)
        # # Returns true if the response doesn't match one of the ignore phrases.
        # #
        # def is_relevant(response: str, api: ModelAPI):
        #     if ignore_keywords is None:
        #         return True
        #     else:
        #         # Either of the ignore keywords with up to 7 paddings on either side (e.g. for a ".")
        #         regex = f"^.{0,7}(?:{'|'.join(ignore_keywords)}).{0,7}$"
        #         return not re.match(regex, response, re.DOTALL)

        call_spec = CallSpec(
            "main",
            [
                PassData(
                    "df",
                    ["data", "single-dim", "pandas"],
                    "{{NAME}} = pd.read_json({{PATH}}); {{NAME}}['time'] = pd.to_datetime({{NAME}}['time'])",
                )
            ],
        )
        ## VERSION WITH PRINT
        # call_spec = CallSpec("main", [PassData("df", ["data", "single-dim", "pandas"], "{{NAME}} = pd.read_json({{PATH}}); print({{NAME}}); {{NAME}}['time'] = pd.to_datetime({{NAME}}['time'])")])
        ## VERSION WITH SUPER PRINT
        # call_spec = CallSpec("main", [PassData("df", ["data", "single-dim", "pandas"], "print('start'); import os; print(os.path.exists({{PATH}})); print('OPENING FILE {{PATH}}'); {{NAME}} = pd.read_json({{PATH}}); print('loaded'); print({{NAME}}); {{NAME}}['time'] = pd.to_datetime({{NAME}}['time'])")])

        prepare_sentences_and_prompts = (
            # 1. Split sentences
            t.SentenceSplit(TEXT, PART),
            t.LimitEntries(4),  # TODO: This is only for debug!!! Remove later!!!
            # 2. Ask for a code
            t.ApplyTemplate(code_start_template, START_WITH),
            t.ApplyTemplate(template_code, CODE_PROMPT),
            t.ApplyTemplate(template_decide, DECIDE_PROMPT),
        )

        coding_loop = CodingLoop(
            decision_prompt=t.ConverseLLM(DECIDE_PROMPT, CodingLoop.FIELD_CONVERSATION),
            code_prompt=t.ConverseLLM(CODE_PROMPT, CodingLoop.FIELD_CONVERSATION, start_with_field=START_WITH),
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
            max_iters=5,
        )

        annotate_and_join = (
            t.ApplyTemplate(template_annotate, ANNOTATION_PROMPT),
            t.ConverseLLM(ANNOTATION_PROMPT, CONVERSATION),
            t.ConversationExtractResponse(CONVERSATION, ANNOTATION_RESPONSE),
            # t.AskPrompt(ANNOTATION_PROMPT, ANNOTATION_RESPONSE),
            t.StringifyConversation(CONVERSATION, CONVERSATION),
            t.Log(text="Conversation... \n", field=CONVERSATION, join_by=t.join_string_long),
            # 4. Ask to annotate code result
            t.Log("Annotation response: ", field=ANNOTATION_RESPONSE, join_by=t.join_string_long),
            t.ExtractTag(
                ANNOTATION_RESPONSE,
                THINKING_TRACE,
                tag="think",
                join_occurances=True,
                remove_from_input=True,
                log_as="THINKING",
            ),
            # 5. Join answers
            # TODO: Unification has to be resolved with conversations somehow. I would rather be able to save the conversations. Maybe add 'conversation_fields' and (Transcribe the conversations? Or make a list of lists?)
            # TODO: Don't ignore the conversation
            t.Unify(annotation_fields=[ANNOTATION_RESPONSE], join_strings_by=t.join_string_long),
            t.ExtractJson(ANNOTATION_RESPONSE, EXTRACTED),
            t.ParseAnnotations(
                EXTRACTED, ANNOTATIONS, annotation_span_categories, annotation_overlap_allowed, output_validation_model
            ),
        )

        return [
            *prepare_sentences_and_prompts,
            coding_loop,
            *annotate_and_join,
            # TODO: Add the whole conversation to metadata.
            t.Metadata(fields=[CODE_PROMPT, ANNOTATION_PROMPT]),
        ]


@register_llm_eval("fact_split_coder_agent")
class CoderAgentStrategy(SequentialStrategy):
    def get_transform_sequence(self) -> list[t.Transform]:
        TEXT = SequentialStrategy.TEXT
        DATA = SequentialStrategy.DATA

        # -> facts
        FACT_PROMPT = "fact_prompt"
        TEXT_WITH_FACT_TAGS = "text_with_fact_tags"
        FACT = "fact"

        DECIDE_PROMPT = "decide_prompt"

        CODE_PROMPT = "code_prompt"
        CODE_RESPONSE = "code_response"
        CODE = "code"
        CONVERSATION = "code_loop_conversation"

        START_WITH = "code_start"
        THINKING_TRACE = "thinking_trace"
        ANNOTATION_PROMPT = "annotation_prompt"
        ANNOTATION_TEXT = "annotation_text"
        ANNOTATION_RESPONSE = "annotation_response"
        ANNOTATIONS = SequentialStrategy.ANNOTATIONS
        EXTRACTED = "extracted"

        super_template = self.config["prompt_template"]
        section_facts = "isolate facts"
        section_decide = "decide action"
        section_write_code = "write code"
        section_code_start = "code start"
        section_annotate = "annotate"
        all_sections = [section_facts, section_decide, section_write_code, section_code_start, section_annotate]

        sections = get_template_sections(super_template, all_sections)

        code_start_template = sections[section_code_start]
        template_facts = sections[section_facts]
        template_code = sections[section_write_code]
        template_annotate = sections[section_annotate]
        template_decide = sections[section_decide]

        ## super_template example:
        # --- decide action ---
        # You are given the following annotation:
        # (...)
        # --- write code ---
        # You are given the following annotation:
        # (...)
        # --- code start ---
        # ```python
        # (...)
        # --- annotate ---
        # Given these results, (...)

        # I could make the Modules return args they need. Possibly they could even be sub-dictionaries.
        annotation_span_categories = self.config["annotation_span_categories"]
        annotation_overlap_allowed = self.config.get("annotation_overlap_allowed", False)
        with_reason = self.extra_args.get("with_reason", True)
        output_validation_model = AnnotationModelFactory.get_output_model(with_reason)

        # ignore_keywords = self.extra_args.get("ignore_keywords", None)
        # # Returns true if the response doesn't match one of the ignore phrases.
        # #
        # def is_relevant(response: str, api: ModelAPI):
        #     if ignore_keywords is None:
        #         return True
        #     else:
        #         # Either of the ignore keywords with up to 7 paddings on either side (e.g. for a ".")
        #         regex = f"^.{0,7}(?:{'|'.join(ignore_keywords)}).{0,7}$"
        #         return not re.match(regex, response, re.DOTALL)

        call_spec = CallSpec(
            "main",
            [
                PassData(
                    "df",
                    ["data", "single-dim", "pandas"],
                    "{{NAME}} = pd.read_json({{PATH}}); {{NAME}}['time'] = pd.to_datetime({{NAME}}['time'])",
                )
            ],
        )

        text = "BTC/GBP on Coinbase Pro for 1-day intervals from March 31, 2024 to March 28, 2025: - **Overall Trend**: The time series shows a general upward trend with significant volatility. Bitcoin's price fluctuates widely over this period but ends higher than it started. - **Key Prices**: - **Opening Price (March 31, 2024)**: £55037 - **Closing Price (March 28, 2025)**: £64972.87 - **Price Extremes**: - **Highest High**: £89499.99 on February 20, 2025 - **Lowest Low**: £38665.31 on August 5, 2024 - **Notable Events**: - A significant peak occurred around January 20, 2025, where the price reached a high of £89499.99. This could be attributed to positive news or increased institutional investment. - A notable dip was observed on August 5, 2024, with the low of £38665.31. This might have been due to regulatory changes, market corrections, or other external economic factors. - **Volatility**: - The series exhibits high volatility, with frequent and large price swings. This is typical for cryptocurrency markets, which are often influenced by speculative trading, news events, and technological developments. - **Stability Periods**: - There were periods of relative stability, particularly around May and June 2024, where the price fluctuated within a narrower range. This time series highlights the dynamic nature of Bitcoin's value in the digital currency market, influenced by various factors including market sentiment, regulatory changes, and macroeconomic conditions."

        text_with_facts = "<fact>BTC/GBP</fact> <fact>on Coinbase Pro</fact> <fact>for 1-day intervals</fact> from <fact>March 31, 2024</fact> to <fact>March 28, 2025</fact>: - **Overall Trend**: <fact>The time series shows a general upward trend</fact> <fact>with significant volatility</fact>. <fact>Bitcoin's price fluctuates widely over this period</fact> but <fact>ends higher than it started</fact>. - **Key Prices**: - **Opening Price** (<fact>March 31, 2024</fact>): <fact>£55037</fact> - **Closing Price** (<fact>March 28, 2025</fact>): <fact>£64972.87</fact> - **Price Extremes**: - **Highest High**: <fact>£89499.99</fact> <fact>on February 20, 2025</fact> - **Lowest Low**: <fact>£38665.31</fact> <fact>on August 5, 2024</fact> - **Notable Events**: - <fact>A significant peak occurred</fact> <fact>around January 20, 2025</fact>, where the price reached a high of <fact>£89499.99</fact>. <fact>This could be attributed to positive news or increased institutional investment</fact>. - <fact>A notable dip was observed</fact> <fact>on August 5, 2024</fact>, with the low of <fact>£38665.31</fact>. <fact>This might have been due to regulatory changes, market corrections, or other external economic factors</fact>. - **Volatility**: - <fact>The series exhibits high volatility</fact>, <fact>with frequent and large price swings</fact>. <fact>This is typical for cryptocurrency markets</fact>, <fact>which are often influenced by speculative trading, news events, and technological developments</fact>. - **Stability Periods**: - <fact>There were periods of relative stability</fact>, <fact>particularly around May and June 2024</fact>, where <fact>the price fluctuated within a narrower range</fact>. <fact>This time series highlights the dynamic nature of Bitcoin's value in the digital currency market</fact>, <fact>influenced by various factors including market sentiment, regulatory changes, and macroeconomic conditions</fact>."

        return [
            t.ApplyTemplate(template_facts, FACT_PROMPT),
            t.AskPrompt(FACT_PROMPT, TEXT_WITH_FACT_TAGS),
            # t.Put(text, TEXT),
            # t.Put(text_with_facts, TEXT_WITH_FACT_TAGS),
            DebugShowFacts(TEXT_WITH_FACT_TAGS, ANNOTATIONS),
            # t.LogAllThrow(),
        ]

        prepare_sentences_and_prompts = (
            # 1. Split sentences
            # t.SentenceSplit(TEXT, PART),
            # t.LimitEntries(3), # TODO: This is only for debug!!! Remove later!!!
            t.ApplyTemplate(template_facts, FACT_PROMPT),
            t.LogAllThrow(),
            # 2. Ask for a code
            t.ApplyTemplate(code_start_template, START_WITH),
            t.ApplyTemplate(template_code, CODE_PROMPT),
            t.ApplyTemplate(template_decide, DECIDE_PROMPT),
        )

        coding_loop = CodingLoop(
            decision_prompt=t.ConverseLLM(DECIDE_PROMPT, CodingLoop.FIELD_CONVERSATION),
            code_prompt=t.ConverseLLM(CODE_PROMPT, CodingLoop.FIELD_CONVERSATION, start_with_field=START_WITH),
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
        )

        annotate_and_join = (
            t.ApplyTemplate(template_annotate, ANNOTATION_PROMPT),
            t.ConverseLLM(ANNOTATION_PROMPT, CONVERSATION),
            t.ConversationExtractResponse(CONVERSATION, ANNOTATION_RESPONSE),
            # t.AskPrompt(ANNOTATION_PROMPT, ANNOTATION_RESPONSE),
            # 4. Ask to annotate code result
            t.Log("Annotation response: ", field=ANNOTATION_RESPONSE, join_by=t.join_string_long),
            t.ExtractTag(
                ANNOTATION_RESPONSE,
                THINKING_TRACE,
                tag="think",
                join_occurances=True,
                remove_from_input=True,
                log_as="THINKING",
            ),
            # 5. Join answers
            # TODO: Unification has to be resolved with conversations somehow. I would rather be able to save the conversations. Maybe add 'conversation_fields' and (Transcribe the conversations? Or make a list of lists?)
            # TODO: Don't ignore the conversation
            t.Unify(
                annotation_fields=[ANNOTATION_RESPONSE],
                join_strings_by=t.join_string_long,
                ignore_fields=[CONVERSATION],
            ),
            t.ExtractJson(ANNOTATION_RESPONSE, EXTRACTED),
            t.ParseAnnotations(
                EXTRACTED, ANNOTATIONS, annotation_span_categories, annotation_overlap_allowed, output_validation_model
            ),
        )

        return [
            *prepare_sentences_and_prompts,
            coding_loop,
            *annotate_and_join,
            # TODO: Add the whole conversation to metadata.
            t.Metadata(fields=[CODE_PROMPT, ANNOTATION_PROMPT]),
        ]
