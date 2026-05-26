#!/usr/bin/env python

from pathlib import Path
from factgenie.llm_judge_model import AnnItem, AnnCategory, Ann, DatasetItem, Evaluator, DatasetInfo, DatasetEval, get_metadata, load_file
from rich import print
from rich.rule import Rule
from rich.table import Table
from typing import Literal, get_args

import numpy as np
import argparse
import pandas as pd


# Args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dir",
    default="judge-outputs",
    type=str,
    help="The directory containing all .jsonl files",
)
parser.add_argument(
    "--output_file",
    "-o",
    default=None,
    type=str,
    help="File to save the aggregated results to.",
)
parser.add_argument(
    "--explanations",
    default=False,
    action="store_true",
    help="Set to true if you want the oiutput csv to contains explanations."
)


def load(dir: Path, path: str):
    """
    Args:
        dir (Path): The directory holding the path. Used for answer finding.
        path (str): Absolute path to the LLM-as-judge jsonl file.
    """

    ds_info, evaluator = get_metadata(path)

    file = load_file(path)
    # merged = merge_files(file)  # Checks that conversations are identical
    merged = file

    dataset_items = [DatasetItem(i, m.conversation.strip(), Ann(m.annotations)) for i, m in enumerate(merged)]

    dataset_eval = DatasetEval(ds_info, [evaluator], dataset_items)

    data = dir / "adding-correctness" / "data"
    split = dataset_eval.info.split.replace("final-", "")
    outputs_file = data / f"{split}.jsonl"
    answers_file = data / f"answers-{split.split('-')[0]}.jsonl"
    dataset_eval.add_correct(str(outputs_file), str(answers_file))

    return dataset_eval


def parse_judges(args):
    dir = Path(__file__).parent / args.dir
    files = list(dir.glob(f"*.jsonl"))

    data_items = []
    for f in files:
        print(f"Loading {f.name}")
        eval = load(dir, str(f))

        # Decode the output source...
        dataset = "tse" if "tse" in eval.info.split else \
                  "tsfu" if "tsfu" in eval.info.split else None
        model = "qwen" if "qwen" in eval.info.split else \
                "gpt-oss" if "gpt-oss" in eval.info.split else None
        strategy = "coder" if "coder" in eval.info.split else \
                   "hybrid" if "hybrid" in eval.info.split else \
                   "direct" if "direct" in eval.info.split else None

        def categories(ex):
            return {
                f"{cat.name}/{item.name}": item.choices[0]
                for cat in ex.annotations.categories
                for item in cat.items
            }

        def explained_categories(ex):
            d = {}
            for cat in ex.annotations.categories:
                for item in cat.items:
                    d[f"{cat.name}/{item.name}"] = item.choices[0]
                    d[f"{cat.name}/{item.name}-explanation"] = item.explanations[0]
            return d

        for ex in eval.examples:
            try:
                data_items.append({
                    "dataset": dataset,
                    "model": model,
                    "strategy": strategy,
                    "example_idx": ex.example_index,
                    "correct": ex.correct,
                    "category": ex.category,
                    **(explained_categories(ex) \
                       if args.explanations \
                       else categories(ex)),
                    # **{
                    #     f"{cat.name}/{item.name}": item.choices[0]
                    #     for cat in ex.annotations.categories
                    #     for item in cat.items
                    # },
                })
            except:
                print(f"Skipping {dataset} {model} {strategy} at index {ex.example_index} (missing output)")
    df = pd.DataFrame(data_items)
    return df


# Entry point
def main(args):
    # LOAD
    df = parse_judges(args)

    # PRINT
    print()
    print(df)
    print()
    print("index:\n •", "\n • ".join(df.columns))

    # SAVE
    if args.output_file is not None:
        print(f"saving to {args.output_file}...", end=" ")
        # df.to_json(args.output_file)
        if not args.output_file.endswith("csv") and not args.output_file.endswith("jsonl"):
            args.output_file += ".csv"

        # Save to csv by default, json if chosen.
        if args.output_file.endswith("jsonl"):
            # with open(args.output_file, "w") as f:
            #     print(df.to_json(orient='records', lines=True))

            df.to_json(args.output_file, orient='records', lines=True)
        else:
            df.to_csv(args.output_file)

        print(f"done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

# ./qa_summary.py -o summary.csv
