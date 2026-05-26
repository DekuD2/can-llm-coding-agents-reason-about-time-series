#!/usr/bin/env python

from factgenie.llm_judge_model import DatasetEval
from rich import print
from rich.rule import Rule
from rich.table import Table
from typing import Literal, get_args
import numpy as np
import pandas as pd

import argparse


# Args
parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="A path to the first jsonl file with answers.")
parser.add_argument("file2", type=str, help="A path to the second jsonl file with answers.")
parser.add_argument(
    "--output_file",
    "-o",
    default=None,
    type=str,
    help="File to save to.",
)
parser.add_argument(
    "--save_csv",
    default=None,
    type=str,
    help="Save results as csv."
)


# Precalculate the option counts for individual categories
def n_options(answer_type: type):
    if answer_type == type(True):
        return 2
    options = get_args(answer_type)
    assert len(options) >= 2
    return len(options)

CoreReasonForAnswer = Literal["code", "raw data", "question format", "other"]
CoreProblemSolvingStrategy = Literal["statistical test", "spectral analysis", "curve fitting", "windowed/rolling statistics", "simple arithmetic", "other"]
CodeResult = Literal["success", "partial failure", "complete failure", "no code"]

option_counts = {
    "conceptual_misunderstanding": n_options(bool),
    "wrong_core_problem_solving_strategy": n_options(bool),
    "wrong_method_within_strategy": n_options(bool),
    "unsupported_assumption": n_options(bool),
    "implementation_errors": n_options(bool),
    "incorrect_result_interpretation": n_options(bool),
    "insufficient_evidence_guess": n_options(bool),
    "code_result": n_options(CodeResult),
    "tool_usage_trouble": n_options(bool),
    "other": n_options(bool),
    "reasoning_answer_mismatch": n_options(bool),
    "reasoning_tool_usage_mismatch": n_options(bool),
    "hallucinated_values_in_reasoning": n_options(bool),
    "core_reason_for_answer": n_options(CoreReasonForAnswer),
    "core_strategy": n_options(CoreProblemSolvingStrategy),
}


def get_evals(dataset_eval: DatasetEval):
    any_missing = False
    all_evals = []
    all_option_counts = []
    examples = dataset_eval.examples
    for i, example in enumerate(examples):
        for category in example.annotations.categories:
            for item in category.items:
                all_option_counts.append([option_counts[item.name] for _ in item.eval])
                all_evals.append(item.eval)
                if len(item.eval) == 0:
                    print(f"{i}th example is missing {item.name} in {category.name}")
                    any_missing = True

    if any_missing:
        exit(0)

    print("No missing annotations found. Proceeding...")

    all_evals = np.array(all_evals)
    # p_correct = all_evals.sum(axis=0) / all_evals.shape[0]
    # all_option_counts = np.array(all_option_counts)

    return all_evals #, p_correct, all_option_counts


# Entry point
def main(args):
    # if args.json:
    #     global print
    #     def print(*args, **kwargs):
    #         pass

    with open(args.file, "r") as f:
        dataset_eval = DatasetEval.from_json(f.read())

    with open(args.file2, "r") as f:
        dataset_eval2 = DatasetEval.from_json(f.read())

    # h1 = human 1, ...
    h1 = get_evals(dataset_eval)
    h2 = get_evals(dataset_eval2)

    h1_agreement = h1.sum(axis=0) / h1.shape[0]
    h2_agreement = h2.sum(axis=0) / h1.shape[0]
    avg_agreement = (h2_agreement + h1_agreement) / 2

    def fmt_percent(n):
        return f"{n * 100:.2f}%"

    def fmt_decimal(n):
        return f"{n:.4f}"

    def calc_kappa(h1, h2):
        p_0 = (h1 == h2).mean()
        h1_p_agree = h1.mean()
        h2_p_agree = h2.mean()
        p_e = h1_p_agree * h2_p_agree + \
              (1 - h1_p_agree) * (1 - h2_p_agree)
        return (p_0 - p_e) / (1 - p_e)

    t = Table(
        "Model",
        "Campaign",
        "Annotator 1 %",
        "Annotator 2 %",
        "Average %",
        # "Kappa"
    )

    data = []
    for i, e in enumerate(dataset_eval2.evaluators):
        t.add_row(
            e.model,
            e.campaign_id,
            fmt_percent(h1_agreement[i]),
            fmt_percent(h2_agreement[i]),
            fmt_percent(avg_agreement[i]),
            # fmt_decimal(calc_kappa(h1[:, i], h2[:, i]))
        )

        model_suffix = ""
        if "gpt-oss" in e.model:
            model_suffix = "-sysmsg" if "sysmsg" in e.campaign_id else "-nosysmsg"
        if "qwen" in e.model:
            model_suffix = "-two-phase" if "two-phase" in e.campaign_id else "-single-phase"

        data.append({
            "model": e.model.replace("openai/", "") + model_suffix,
            "accuracy": avg_agreement[i]
        })
    print(t)
    print(Rule("inter-annotator agreement"))

    p_0 = (h1 == h2).mean()
    h1_p_agree = h1.mean()
    h2_p_agree = h2.mean()
    p_e = h1_p_agree * h2_p_agree + \
          (1 - h1_p_agree) * (1 - h2_p_agree)
    kappa = (p_0 - p_e) / (1 - p_e)

    print(f"Annotator 1's probability of agreeing with the judge = {fmt_percent(h1_p_agree)} | ({args.file})")
    print(f"Annotator 2's probability of agreeing with the judge = {fmt_percent(h2_p_agree)} | ({args.file2})")
    print(f"p_0 = {fmt_percent(p_0)}, p_e = {fmt_percent(p_e)}")
    print(f"Cohen's kappa = {fmt_decimal(kappa)}")

    if args.save_csv is not None:
        df = pd.DataFrame(data)
        df.to_csv(args.save_csv)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
