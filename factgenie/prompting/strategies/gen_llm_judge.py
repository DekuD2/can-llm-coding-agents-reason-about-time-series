from typing import Literal
from pydantic import BaseModel

from factgenie.prompting.strategies import register_llm_gen, SequentialStrategy
from factgenie.prompting import transforms as t
from factgenie.prompting.experimental_transforms import DataPathAsMetadata

import logging
logger = logging.getLogger("factgenie")


CoreReasonForAnswer = Literal["code", "raw data", "question format", "other"]
CoreProblemSolvingStrategy = Literal["statistical test", "spectral analysis", "curve fitting", "windowed/rolling statistics", "simple arithmetic", "other"]
CodeResult = Literal["success", "partial failure", "complete failure", "no code"]

class MethodologicalProblems(BaseModel):
    conceptual_misunderstanding: bool
    conceptual_misunderstanding_explanation: str | None

    wrong_core_problem_solving_strategy: bool
    wrong_core_problem_solving_strategy_explanation: str | None

    wrong_method_within_strategy: bool
    wrong_method_within_strategy_explanation: str | None

    unsupported_assumption: bool
    unsupported_assumption_explanation: str | None

    implementation_errors: bool
    implementation_errors_explanation: str | None

    incorrect_result_interpretation: bool
    incorrect_result_interpretation_explanation: str | None

    insufficient_evidence_guess: bool
    insufficient_evidence_guess_explanation: str | None

class CodeProblems(BaseModel):
    code_result: CodeResult
    code_result_explanation: str | None

    tool_usage_trouble: bool
    tool_usage_trouble_explanation: str | None

    other: bool
    other_explanation: str | None

class OtherProblems(BaseModel):
    reasoning_answer_mismatch: bool
    reasoning_answer_mismatch_explanation: str | None

    reasoning_tool_usage_mismatch: bool
    reasoning_tool_usage_mismatch_explanation: str | None

    hallucinated_values_in_reasoning: bool
    hallucinated_values_in_reasoning_explanation: str | None

class ErrorAnalysis(BaseModel):
    core_reason_for_answer: CoreReasonForAnswer
    core_reason_for_answer_explanation: str

    core_strategy: CoreProblemSolvingStrategy
    core_strategy_explanation: str

    methodological_problems: MethodologicalProblems
    code_problems: CodeProblems
    other_problems: OtherProblems


@register_llm_gen(name="llm_judge")
class LLMJudgeStrategy(SequentialStrategy):
    def get_transform_sequence(self) -> list[t.Transform]:
        PROMPT = "prompt"
        OUTPUT = SequentialStrategy.OUTPUT

        ORIG_CONV = "agent_conversation"
        CONVERSATION = "conversation"

        system_msg = self.config.get("system_msg", None)
        starts_with = self.config.get("start_with", None)

        extra_args = self.config.get("extra_args", {})
        stopping_sequence = self.extra_args.get("stopping_sequence", None)
        remove_suffix = self.extra_args.get("remove_suffix", None)

        model_prepends = extra_args.get("model_prepends", None)
        thought_context = extra_args.get("thought_context", False)
        two_phase = extra_args.get("two_phase", False)

        if two_phase:
            logger.info("Two phase detected!")

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

        api_provider = self.config.get("api_provider", "").lower()

        import litellm
        if litellm.enable_json_schema_validation != True:
            logger.warning("Setting `litellm.enable_json_schema_validation = True` for the LLMJudgeStrategy!")
            litellm.enable_json_schema_validation = True

        if two_phase: 
            prompting = [
                t.ConverseLLM(
                    PROMPT, 
                    CONVERSATION,
                    system_msg=system_msg,
                    start_with=starts_with,
                    extractors=t.ConverseLLM.EXTRACTORS_LOG_THINKING,
                    # ensure_completion=True,
                    model_prepends=model_prepends,
                    thought_context=thought_context,
                ),
                t.Put("Please write your answer again inside a structured output.", "prompt_2"),
                t.ConverseLLM(
                    "prompt_2",
                    CONVERSATION,
                    system_msg=system_msg,
                    start_with=starts_with,
                    extractors=t.ConverseLLM.EXTRACTORS_LOG_THINKING,
                    ensure_completion=True,
                    # model_prepends=model_prepends,
                    thought_context=thought_context,
                    completion_kwargs={"think": True, "response_format": ErrorAnalysis},
                ),
                t.ConversationExtractResponse(CONVERSATION, OUTPUT),
            ]
        elif "openai" in api_provider:
            prompting = [
                # I am not sure if we should prepend thinking here, but it seems to work?
                t.AskPrompt(PROMPT, OUTPUT, system_msg, starts_with, completion_kwargs={"response_format": ErrorAnalysis}),
            ]
        else:
            prompting = [
                # I am not sure if we should prepend thinking here, but it seems to work?
                t.AskPrompt(PROMPT, OUTPUT, system_msg, starts_with, completion_kwargs={"think": True, "response_format": ErrorAnalysis}),
            ]

        return [
            t.ApplyTemplate(self.config["prompt_template"], PROMPT),

            *prompting,

            # t.AskPrompt(PROMPT, OUTPUT, system_msg, starts_with, completion_kwargs={"think": True, "response_format": ErrorAnalysis}, model_prepends=model_prepends),

            t.PostprocessOutput(OUTPUT, OUTPUT, stopping_sequence, remove_suffix),
            # Logging and metadata.
            t.Log(text="Output: ", field=OUTPUT),
            # Put the original conversation to make the job easier.
            t.ApplyTemplate("{data[conv]}", ORIG_CONV),
            # Should contain the following keys: "model", "prompt_strat", "prompt_template", "system_msg", "model_args", "extra_args", "annotator_id", "campaign_id"
            # (I.e. {"metadata": {"original_metadata": {"model": ..., "prompt_strat": ..., ...}}})
            DataPathAsMetadata(["data", "original_metadata"], "original_metadata"),
            t.Metadata(fields=[PROMPT, ORIG_CONV]),
        ]

