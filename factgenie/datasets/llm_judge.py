import json
import logging
import numpy as np
import os
import pandas as pd
import plotly.express as px
import markdown

from factgenie.datasets.time_series_exam import TimeSeriesExam
from factgenie.datasets.time_series_feature_understanding import TimeSeriesFeatureUnderstanding

logger = logging.getLogger("factgenie")

from datasets import load_dataset
from math import isinf, isnan
from pathlib import Path
from time import perf_counter

from factgenie.datasets.dataset import Dataset
from factgenie.datasets.utils import extract_conversation, tree


file_names = {
    "tse-hybrid": "tse-hybrid-minimal.jsonl",
    "tse-tools": "tse-tools.jsonl",
    "tse-raw": "tse-raw.jsonl",
    "tsfu-hybrid": "tsfu-hybrid-minimal.jsonl",
    "tsfu-raw": "tsfu-raw.jsonl",
    # Final:
    "final-tse-gpt-oss-coder": "final/tse-gpt-oss-coder-th30k.jsonl",
    "final-tse-gpt-oss-direct": "final/tse-gpt-oss-direct-th30k.jsonl",
    "final-tse-gpt-oss-hybrid": "final/tse-gpt-oss-hybrid-th30k.jsonl",
    "final-tse-qwen-coder": "final/tse-qwen-coder-th82k-cot.jsonl",
    "final-tse-qwen-direct": "final/tse-qwen-direct-th82k-cot.jsonl",
    "final-tse-qwen-hybrid": "final/tse-qwen-hybrid-th82k-cot.jsonl",
    "final-tsfu-gpt-oss-coder": "final/tsfu-gpt-oss-coder.jsonl",
    "final-tsfu-gpt-oss-direct": "final/tsfu-gpt-oss-direct.jsonl",
    "final-tsfu-gpt-oss-hybrid": "final/tsfu-gpt-oss-hybrid.jsonl",
}


answer_files = {
    "tse-hybrid": "answers-tse-test.jsonl",
    "tse-tools": "answers-tse-test.jsonl",
    "tse-raw": "answers-tse-test.jsonl",
    "tsfu-hybrid": "answers-tsfu-test.jsonl",
    "tsfu-raw": "answers-tsfu-test.jsonl",
    # Final:
    "final-tse-gpt-oss-coder": "final/answers-tse.jsonl",
    "final-tse-gpt-oss-direct": "final/answers-tse.jsonl",
    "final-tse-gpt-oss-hybrid": "final/answers-tse.jsonl",
    "final-tse-qwen-coder": "final/answers-tse.jsonl",
    "final-tse-qwen-direct": "final/answers-tse.jsonl",
    "final-tse-qwen-hybrid": "final/answers-tse.jsonl",
    "final-tsfu-gpt-oss-coder": "final/answers-tsfu.jsonl",
    "final-tsfu-gpt-oss-direct": "final/answers-tsfu.jsonl",
    "final-tsfu-gpt-oss-hybrid": "final/answers-tsfu.jsonl",
}


# Puts conversation into "conv"
class LLMJudgeDataset(Dataset):
    MINI_SPLIT_DATASET_LEN = 50
    test_ds = None

    def __init__(self, *vargs, **kwargs):
        self.example_shown = False
        super().__init__(*vargs, **kwargs)

    def load_examples(self, split: str, data_path):
        self.datasplit_db = {}

        self.data_path = data_path

        some_correct_found = False

        examples = []
        data_path = Path(data_path)
        file = data_path / file_names.get(split, f"{split}.jsonl")
        answer_file = data_path / answer_files.get(split, f"{split}-answers.jsonl")
        logger.info(f"answers file: {answer_file}")
        assert answer_file.exists(), f"Answers file {answer_file} doesn't exist!"
        idx = 0
        with file.open() as f:
            with answer_file.open() as af:
                for pred_line, golden_line in zip(f.readlines(), af.readlines()):
                    pred = json.loads(pred_line)
                    golden = json.loads(golden_line)

                    # Given this LLM's conversation, where did it make an error? The correct answer was supposed to be {data[correct]}.
                    #
                    # ===
                    # {data[conv]}
                    # ===

                    copy_metadata = {"model", "prompt_strat", "prompt_template", "system_msg", "model_args", "extra_args", "annotator_id", "campaign_id"}
                    copy_data = {"dataset", "split", "example_idx"}
                    item = {
                        "conv": extract_conversation(pred),
                        "correct": golden["answer"],
                        "selected": pred['options'][0]['value'][3:],
                        "original_index": idx,
                        "split": split,
                        "original_metadata": {k: v for k, v in pred["metadata"].items() if k in copy_metadata},
                        "orig_data": {k: v for k, v in pred.items() if k in copy_data}
                    }

                    examples.append(item)
                    if item["correct"] == item["selected"]:
                        some_correct_found = True

                    idx += 1

        if not some_correct_found:
            logger.warning("No correct answers were found! The code in `datasets/llm-judge.py` is likely wrong.")

        logger.info(f"Split {split} loaded with {len(examples)} examples. Idx = {idx}")
        return examples

    def get_image_smart(self, example):
        # from rich import print
        # print(list(example.keys()))
        orig_data = example["orig_data"]
        dataset = orig_data["dataset"]
        split = orig_data["split"]
        example_idx = orig_data["example_idx"]

        key = (dataset, split)
        if key not in self.datasplit_db:
            if dataset == "time-series-exam":
                ds = TimeSeriesExam(dataset_id=f"{dataset} for error types")
            elif dataset == "time-series-feature-understanding":
                ds = TimeSeriesFeatureUnderstanding(dataset_id=f"{dataset} for error types")
            else:
                raise NotImplementedError(f"Unknown dataset '{dataset}'")
            ds_path = Path(self.data_path) / ".." / dataset  # not used anyways
            examples = ds.load_examples(split, str(ds_path))
            self.datasplit_db[key] = (ds, examples)
            del ds, examples

        ds, examples = self.datasplit_db[key]
        logger.info(f"returning figs for tse example {example_idx}")
        return ds.render_figs(examples[example_idx])
        

    def render(self, example):
        fig_htmls = self.get_image_smart(example)
        # fig_htmls = ["<p>" + example["conv"] + "</p>"]
        fig_htmls.append(markdown.markdown(example["conv"].replace("\n", "\n\n"), extensions=["markdown.extensions.tables"]))

        # markdown.markdown(example["conv"], extensions=["markdown.extensions.tables"])
        html = ""
        # html = markdown.markdown(example["conv"], extensions=["markdown.extensions.tables"])

        return (
            """<div id="graph">"""
            + "\n".join(fig_htmls)
            + """</div><div class="root" style="margin-top: 40px">"""
            + html
            + """</div>"""
        )
