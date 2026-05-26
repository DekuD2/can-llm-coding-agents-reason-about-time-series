import json
import logging
import numpy as np
import os
import pandas as pd
import plotly.express as px

logger = logging.getLogger("factgenie")

from datasets import load_dataset
from math import isinf, isnan
from pathlib import Path
from time import perf_counter

from factgenie.datasets.dataset import Dataset
from factgenie.datasets.utils import deep_round, deep_round_v2, tree


class TimeSeriesExam(Dataset):
    MINI_SPLIT_DATASET_LEN = 50
    test_ds = None

    def __init__(self, *vargs, **kwargs):
        self.example_shown = False
        super().__init__(*vargs, **kwargs)

    def load_examples(self, split: str, data_path):
        assert split in ["test", "mini"], f"Unsupported split '{split}'."
        start = perf_counter()

        if TimeSeriesExam.test_ds is None:
            TimeSeriesExam.test_ds = load_dataset("AutonLab/TimeSeriesExam1")["test"]
        test = TimeSeriesExam.test_ds

        # TODO: Fix the problem with time series not being the smae length.
        # HOW: It's okay if they are not the same length. Just load them as separate data frames? (but then how to pass along different number of parameters each time... a dict of data frames?)

        # TODO: After that, create a simple passthru strategy to just extract the question into whatever pattern we want to ask (with or without hints).
        examples = []
        answers = []
        for i, item in enumerate(test):
            if split == "mini" and i >= self.MINI_SPLIT_DATASET_LEN:
                break

            dict_of_timeseries = {}
            if item["ts"] is not None:
                dict_of_timeseries |= {"ts": item["ts"]}
            if item["ts1"] is not None:
                dict_of_timeseries |= {"ts1": item["ts1"]}
            if item["ts2"] is not None:
                dict_of_timeseries |= {"ts2": item["ts2"]}

            keys = list(dict_of_timeseries.keys())
            keys_ticks = list(map(lambda x: f"'{x}'", keys))
            keys_backticks = list(map(lambda x: f"`{x}`", keys))
            keys_backticks_def = list(map(lambda x: f"`{x}: pd.DataFrame`", keys))

            item["timeseries"] = dict_of_timeseries
            item["json"] = json.dumps(dict_of_timeseries)

            # item["json_rounded_1"] = json.dumps(deep_round(dict_of_timeseries, digits=1))
            json_rounded_2 = deep_round(dict_of_timeseries, digits=2)
            item["json_rounded_2"] = json.dumps(json_rounded_2)
            # item["json_rounded_3"] = json.dumps(deep_round(dict_of_timeseries, digits=3))
            item["json_keys"] = ", ".join(keys_ticks)
            item["json_keys_raw"] = ", ".join(dict_of_timeseries.keys())

            v2_rounded_2: dict[str, list[float | int]] = deep_round_v2(dict_of_timeseries, digits=2)
            v2_rounded_3: dict[str, list[float | int]] = deep_round_v2(dict_of_timeseries, digits=3)
            v2_rounded_4: dict[str, list[float | int]] = deep_round_v2(dict_of_timeseries, digits=4)
            item["v2_rounded_2"] = json.dumps(v2_rounded_2)
            item["v2_rounded_3"] = json.dumps(v2_rounded_3)
            item["v2_rounded_4"] = json.dumps(v2_rounded_4)

            def format_values(values: list[float|int]):
                return "\n".join(f"idx: {i}, value: {v}" for i, v in enumerate(values))

            def format_whole(rounded_dict: dict[str, list[float|int]]):
                return "\n\n".join(f"{k}:\n{format_values(v)}" for k, v in rounded_dict.items())

            item["v2_rounded_2_fmt"] = format_whole(v2_rounded_2)
            item["v2_rounded_3_fmt"] = format_whole(v2_rounded_3)
            item["v2_rounded_4_fmt"] = format_whole(v2_rounded_4)

            item["globals_rounded_2"] = "\n\n".join(f"Preview of {key}: {', '.join(map(str, value))}" for key, value in json_rounded_2.items())

            if len(dict_of_timeseries) == 1:
                item["json_keys_grammatical"] = keys[0]
            else:
                item["json_keys_grammatical"] = ", ".join(keys[:-1]) + " and " + keys[-1]

            if len(dict_of_timeseries) == 1:
                item["json_keys_grammatical_backticks"] = keys_backticks[0]
            else:
                item["json_keys_grammatical_backticks"] = ", ".join(keys_backticks[:-1]) + ", and " + keys_backticks[-1]

            if len(dict_of_timeseries) == 1:
                item["json_keys_grammatical_backticks_def"] = keys_backticks_def[0]
            else:
                item["json_keys_grammatical_backticks_def"] = ", ".join(keys_backticks_def[:-1]) + ", and " + keys_backticks_def[-1]

            item["json_first_key"] = keys[0]

            item["globals_definitions"] = "\\n".join(map(lambda x: f"{x}: pd.DataFrame  # contains a single unnamed column", keys))

            item["plural"] = "s" if len(dict_of_timeseries) > 1 else ""

            # Before prepending 'A)', 'B)', etc., figure out the answer index.
            item["answer_index"] = item["options"].index(item["answer"])
            answers.append(
                {
                    "type": "multiple_choice_question_answering",
                    "example_idx": i,
                    "quesiton": item["question"],
                    "options": item["options"],
                    "answer": item["answer"],
                    "answer_index": item["answer_index"],
                    "category": item["category"],
                }
            )

            # Prepend options with 'A)', 'B)', etc.
            # We want these modified options to be the 'official' options used in factgenie system..
            item["options"] = [f"{chr(ord('A') + i)}) {option}" for i, option in enumerate(item["options"])]
            item["A)options"] = "\n".join(" " + option for option in item["options"])

            # # DEBUG print an example structure.
            # if not self.example_shown:
            #     self.example_shown = True
            #     logger.info(
            #         "Time Series Exam example structure (either ts is a list or ts1 and ts2 are lists):" + tree(item)
            #     )

            examples.append(item)

        answers_file = Path(data_path) / f"{split}-answers.jsonl"
        if not answers_file.exists():
            logger.info(f"Creating answers file for the QA task. File location: '{answers_file}'")

            os.makedirs(data_path, exist_ok=True)
            with open(answers_file, "w") as f:
                f.writelines(json.dumps(a) + "\n" for a in answers)

        end = perf_counter()
        # logger.info(f"Loading of TSE:{split} ({len(examples)} items) took {end - start:.1f} seconds.")

        return examples

    def render_figs(self, example):
        fig_htmls = []
        for key, data in example["timeseries"].items():
            df = pd.DataFrame({"item": range(len(data)), "value": data})

            fig = px.line(
                df,
                x="item",
                y="value",
                title=key,
                template="plotly_white",
                # hover_data=["time", *features.keys()]
            )
            fig_htmls.append(fig.to_html(include_plotlyjs="cdn"))

        return fig_htmls
        
    def render(self, example):
        # fig_htmls = []
        # for key, data in example["timeseries"].items():
        #     df = pd.DataFrame({"item": range(len(data)), "value": data})

        #     fig = px.line(
        #         df,
        #         x="item",
        #         y="value",
        #         title=key,
        #         # hover_data=["time", *features.keys()]
        #     )
        #     fig_htmls.append(fig.to_html(include_plotlyjs="cdn"))

        fig_htmls = self.render_figs(example)

        html = ""

        return (
            """<div id="graph">"""
            + "\n".join(fig_htmls)
            + """</div><div class="root" style="margin-top: 40px">"""
            + html
            + """</div>"""
        )
