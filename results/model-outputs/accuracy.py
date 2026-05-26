#!/usr/bin/env python

import builtins
import argparse
from ctypes import ArgumentError
import json
from os.path import isdir
import numpy as np
import pyperclip

from dataclasses import dataclass
from itertools import repeat, chain
from pathlib import Path
from rich import print
from rich.rule import Rule
from rich.table import Table
from typing import Iterable

from factgenie.prompting.experimental_transforms import ParseRanges
from factgenie.prompting.model_apis import EmptyAPI
# from factgenie.iaa.affiliation_metrics.metrics import pr_from_events


def stringify_conv(conversation: list):
    ROLE = "role"
    CONTENT = "content"
    THINKING = "reasoning_content"
    TOOL_CALLS = "tool_calls"
    FUNCTION = "function"
    NAME = "name"
    ARGUMENTS = "arguments"

    def thinking_if_exists(conv_item: dict):
        thinking: str = conv_item.get(THINKING, "").strip()
        if thinking is None:
            return ""
        if len(thinking) > 1:
            return "<💭>" + thinking + "</💭>\n"  # 🤔💭
        else:
            return ""

    def tool_calls_if_exists(conv_item: dict):
        tool_calls = conv_item.get(TOOL_CALLS, None)
        if tool_calls is None:
            return ""
        else:
            first: dict = tool_calls[0][FUNCTION]
            name = first.get(NAME, "unnamed")
            args_str: str = first[ARGUMENTS]
            try:
                args = json.loads(args_str)
                if isinstance(args, str):
                    first_arg = args
                else:
                    assert len(args) == 1, "Can't stringify multiple args yet"
                    first_arg = list(args.values())[0]
            except json.JSONDecodeError:
                first_arg = args_str
            return f"<🧰 {name}>" + first_arg + f"</🧰 {name}>\n"  # 🤔💭

    return "\n\n".join(
        f"[{conv_item[ROLE]}]\n{thinking_if_exists(conv_item)}{conv_item[CONTENT]}{tool_calls_if_exists(conv_item)}" for conv_item in conversation
    )


parser = argparse.ArgumentParser()
parser.add_argument("answers_file", type=str, default=None, help="A path to the campaign jsonl file with answers.")
parser.add_argument(
    "--make_csv",
    default=False,
    action="store_true",
    help="Make CSV out of the directory.",
)
parser.add_argument(
    "--json",
    default=False,
    action="store_true",
    help="Output a json.",
)

parser.add_argument("--versus", default=None, type=str, help="A second file for comparative analysis (when included).")
parser.add_argument(
    "--output_file",
    default=None,
    type=str,
    help="File to write the answers from our primary models and how it fares against the other model.",
)
parser.add_argument(
    "--output_filter",
    default=None,
    type=str,
    choices=["TF", "FT", "FF", "TT", "T", "F"],
    help="TF = first model is correct, second (--versus) is incorrect. For one model it's just one letter."
)

parser.add_argument(
    "--keys",
    default=False,
    action="store_true",
    help="Print just the keys to help understanding the files' structures.",
)

parser.add_argument(
    "--binary", "-b",
    default=False,
    action="store_true",
    help="Show 'TFs'",
)

parser.add_argument(
    "--interactive", "-i",
    default=False,
    action="store_true",
    help="Interactive inspector.",
)


def debug_print_keys(golden: dict, predicted: dict):
    print(Rule())
    print(f"golden: {golden.keys()}")
    print()
    print()
    print(f"campaign: {predicted.keys()}")
    print()
    print(f"campaign['options'][0]: {predicted['options'][0].keys()}")
    print()
    print(f"campaign['metadata']: {predicted['metadata'].keys()}")
    print(Rule())


def get_where_correct(tse_pred: str, tse_golden: str, show_keys_only: bool = False, max: int | None = None) -> tuple[list[bool], dict[str, list[bool]], dict[str, list[tuple[bool, bool]]]]:
    """
    Args:
        tse_pred: The path to the predictions file.
        tse_golden: The path to the golden data.
    """
    correct = []
    correct_by_category = {}
    detection_by_category = {}

    with open(tse_golden, "r") as golden:
        with open(tse_pred, "r") as pred:
            for g, p in zip(golden.readlines(), pred.readlines()):
                if max is not None and len(correct) >= max:
                    break

                gj = json.loads(g)
                pj = json.loads(p)
                if show_keys_only:
                    debug_print_keys(gj, pj)
                    exit(0)

                # Make sure the alignment is correct...
                data_idx_golden = gj["example_idx"]
                data_idx_pred = pj["example_idx"]
                assert data_idx_golden == data_idx_pred

                correct_idx = gj["answer_index"]
                selected_idx = pj["options"][0]["index"]

                correct.append(correct_idx == selected_idx)

                category = gj["category"]

                if category not in correct_by_category:
                    correct_by_category[category] = []
                correct_by_category[category].append(correct_idx == selected_idx)

                n_options = len(gj["options"])
                golden_detect = correct_idx != n_options - 1
                pred_detect = selected_idx != n_options - 1
                if category not in detection_by_category:
                    detection_by_category[category] = []
                detection_by_category[category].append((pred_detect, golden_detect))

                # print(f"correct: {correct_idx == selected_idx} golden: {correct_idx} correct: {selected_idx}")

    return correct, correct_by_category, detection_by_category


@dataclass
class Example:
    question: str
    options: list[str]
    correct: str
    category: str
    first_correct: bool
    secnd_correct: bool
    first_select: str
    secnd_select: str
    first_conversation: str
    secnd_conversation: str
    has_secnd: bool

    @property
    def options_string(self):
        return "\n".join(map(lambda x: " • " + str(x), self.options))

    def show(self):
        # table = Table(*(["First", "Second"] if self.has_secnd else ["First"]), show_lines=True)
        table = Table()
        table.add_column("First", vertical="bottom")
        if self.has_secnd:
            table.add_column("Second", vertical="bottom")

        first_color = "[green]" if self.first_correct else "[red]"
        secnd_color = "[green]" if self.secnd_correct else "[red]"

        convs = [first_color + self.first_conversation.replace("[", "\\[")]
        if self.has_secnd:
            convs.append(secnd_color + self.secnd_conversation.replace("[", "\\["))
        table.add_row(*convs)

        selects = [first_color + self.first_select]
        if self.has_secnd:
            selects.append(secnd_color + self.secnd_select)
        table.add_row(*selects)

        print(table)
        print(f"Answer: [green]{self.correct}")
        print(f"Category: [blue]{self.category}")


def extract_conversation(d):
    if 'code_loop_conversation' in d['metadata']:
        conv = d['metadata']['code_loop_conversation']
    elif 'conversation' in d['metadata']:
        conv = d['metadata']['conversation']
    else:
        raise ValueError("Can't find any known converstaion field.")
    if isinstance(conv, list):
        conv = stringify_conv(conv)
    return conv

def get_examples(tse_pred: str, tse_golden: str, tse_vs: str | None = None, show_keys_only: bool = False):
    """
    Args:
        tse_pred: The path to the predictions file.
        tse_golden: The path to the golden data.
        tse_vs: The path of comparison data.
    """
    all_examples: list[Example] = []
    
    with open(tse_golden, "r") as golden:
        golden_lines = golden.readlines()
    with open(tse_pred, "r") as pred:
        pred_lines = pred.readlines()

    if tse_vs is not None:
        with open(tse_vs, "r") as vs_pred:
            vs_lines = vs_pred.readlines()
    else:
        vs_lines = repeat('{"metadata": {"conversation": ""}, "options": [{"value": "", "index": 0}]}')

    for g, p1, p2 in zip(golden_lines, pred_lines, vs_lines):
        gj = json.loads(g)
        pj1 = json.loads(p1)
        pj2 = json.loads(p2)

        # Make sure the alignment is correct...
        assert gj["example_idx"] == pj1["example_idx"]
        assert tse_vs is None or gj["example_idx"] == pj2["example_idx"]

        correct_idx = gj["answer_index"]
        selected_idx1 = pj1["options"][0]["index"]
        selected_idx2 = pj2["options"][0]["index"]

        current = Example(
            question=gj["quesiton"],  # (typo in name sadly)
            category=gj["category"],
            correct=gj["answer"],
            options=gj["options"],
            first_correct=correct_idx == selected_idx1,
            secnd_correct=correct_idx == selected_idx2,
            first_select=pj1['options'][0]['value'][3:],
            secnd_select=pj2['options'][0]['value'][3:],
            first_conversation=extract_conversation(pj1),
            secnd_conversation=extract_conversation(pj2),
            has_secnd=tse_vs is not None,
        )

        all_examples.append(current)

    return all_examples


def get_expected_correct(tse_golden: str, num: int) -> list[bool]:
    """
    Args:
        tse_pred: The path to the predictions file.
        tse_golden: The path to the golden data.
    """

    exp_correct = []
    total_exp_per_cat = {}
    total_per_cat = {}

    with open(tse_golden, "r") as golden:
        for i, g in zip(range(num), golden.readlines()):
            gj = json.loads(g)

            # Make sure the alignment is correct...
            exp_correct.append(1 / len(gj["options"]))

            cat = gj["category"]
            if cat not in total_exp_per_cat:
                total_exp_per_cat[cat] = 0
            if cat not in total_per_cat:
                total_per_cat[cat] = 0

            total_exp_per_cat[cat] += 1 / len(gj["options"])
            total_per_cat[cat] += 1

    print("Expected per categories...", end=" ")
    for key, val in total_exp_per_cat.items():
        print(f"{key}: {100 * val / total_per_cat[key]:.1f}%", end=", ")
    print()

    return exp_correct


def save_comparison(
    tse_pred: str, tse_golden: str, tse_theirs: str | None, ours: list[bool], theirs: list[bool] | repeat | None, save_to: str, filter: str | None,
):
    if theirs is None:
        theirs = repeat(None)

    def tf_letters(corr_ours: bool, corr_theirs: bool | None):
        our_letter = "T" if corr_ours else "F"
        their_letter = "" if corr_theirs is None else \
                       "T" if corr_theirs else "F"
        return our_letter + their_letter

    with open(save_to, "w") as f:
        f.write(f"our file: {tse_pred}\ngolden data: {tse_golden}\nversus: {tse_theirs}\n\n\n")
        with open(tse_golden, "r") as golden:
            with open(tse_pred, "r") as pred:
                for g, p, our, their in zip(golden.readlines(), pred.readlines(), ours, theirs):
                    tf = tf_letters(our, their)

                    # We only want the cases that match the filter.
                    if filter is not None and tf != filter:
                        continue

                    # golden:
                    # ['example_idx', 'quesiton', 'options', 'answer', 'answer_index']
                    # campaign:
                    # ['dataset', 'split', 'example_idx', 'output', 'metadata', 'setup_id', 'annotations', 'flags', 'options', 'sliders', 'text_fields']
                    # campaign['options'][0]:
                    # ['label', 'index', 'value', 'optionList']
                    # campaign['metadata']:
                    # ['api_provider', 'model', 'prompt_strat', 'prompt_template', 'system_msg', 'annotation_overlap_allowed', 'annotation_granularity', 'api_url', 'model_args', 'extra_args', 'annotation_span_categories', 'code_loop_conversation', 'annotator_id', 'annotator_group', 'campaign_id', 'start_timestamp', 'end_timestamp']
                    gj = json.loads(g)
                    pj = json.loads(p)
                    conv = extract_conversation(pj)

                    text = "━" * 80
                    text += f"\nourtheir = {tf} ["
                    text += f"\n\nconversation:\n{conv}"
                    text += "\n" + "─" * 10
                    text += f"\ninterpreted as: {pj['options'][0]['value'][3:]}"
                    text += f"\ncorrect answer: {gj['answer']}"
                    text += f"\n\nquestion: {gj['quesiton']}"
                    options_joined = "\n".join(map(lambda x: " • " + str(x), gj["options"]))
                    text += f"\n\nanswers: \n{options_joined}"
                    text += f"\n\nexample idx: \n{gj['example_idx']}"
                    text += "\n" + "─" * 10
                    text += f"\nourtheir = {tf} ]"
                    f.write(text + "\n\n\n")

def infer_dataset(pred_file):
    with open(pred_file, "r") as f:
        line = f.readline()
        j = json.loads(line)
        dataset = j["dataset"]
        split = j["split"]

    script_dir = Path(__file__).parent
    while len(list(script_dir.glob("pyproject.toml"))) == 0:
        if script_dir == "/":
            raise FileNotFoundError("Could't find the project directory, containing the `pyproject.toml`.")
        script_dir = script_dir.parent
    answers = script_dir / "factgenie" / "data" / "inputs" / dataset / f"{split}-answers.jsonl"
    return dataset, split, answers


def infer_name_infos(pred_file):
    with open(pred_file, "r") as f:
        line = f.readline()
        j = json.loads(line)
        meta = j["metadata"]
        campaign = meta["campaign_id"]
        strat = meta["prompt_strat"]
        model = meta["model"]

    return campaign, strat, model


def cat_name(cat: str) -> str:
    if cat == "trend":
        return "Trend"
    elif cat == "seasonality":
        return "Seasonality"
    elif cat == "outliers":
        return "Anomalies"
    elif cat == "volatility":
        return "Volatility"
    elif cat == "structural_break":
        return "Structural Break"
    elif cat == "statistical_property":
        return "Fat Tail"
    elif cat == "stationarity":
        return "Stationarity"
    elif cat == "correlation":
        return "Fixed Correlation"
    elif cat == "lagged_correlation":
        return "Lagged Correlation" # "Lagged (and maybe changing) correlation"
    else:
        return cat

def get_anomaly_scores(tsa_pred: str, tsa_golden: str, show_keys_only: bool = False) -> dict[str, list[float]]:
    """
    Args:
        tse_pred: The path to the predictions file.
        tse_golden: The path to the golden data.
    """
    f1s_by_category: dict[str, list[float]] = {}

    parser = ParseRanges("input", "output")
    empty_api = EmptyAPI()

    with open(tsa_golden, "r") as golden:
        with open(tsa_pred, "r") as pred:
            for g, p in zip(golden.readlines(), pred.readlines()):
                gj = json.loads(g)
                pj = json.loads(p)
                if show_keys_only:
                    debug_print_keys(gj, pj)
                    exit(0)

                # Make sure the alignment is correct...
                data_idx_golden = gj["example_idx"]
                data_idx_pred = pj["example_idx"]
                assert data_idx_golden == data_idx_pred

                gold_ranges = [(a, b) for (a, b) in gj["answer"]]
                # Seems like I didn't save it properly, but whatever...
                pred_ranges_str = pj["output"]
                pred_ranges = parser.parse_ranges({"input": pred_ranges_str}, empty_api)["output"]
                pred_ranges = sorted(pred_ranges, key=lambda x: x[0])
                # TODO: I did a quick fix here for point predictions. Might not be what we want. We might want it symmetrical for example.
                pred_ranges = [(a, b + 1) if a == b else (a, b) for (a, b) in pred_ranges]
                ts_length = gj["ts_length"]
                t_range = (0, ts_length)

                category = gj["category"]

                if len(gold_ranges) == 0:
                    # TODO:
                    # There are a few options:
                    # Always return 0 (But then what else is the classifier supposed to do? Doesn't seem fair.)
                    # Return 1 when it's correct and 0 when it's not correct (Seems most fair currently.)
                    # Skip the example.
                    if len(pred_ranges) == 0:
                        f1 = 1.0
                    else:
                        f1 = 0.0
                else:
                    try:
                        pr = pr_from_events(pred_ranges, gold_ranges, t_range)
                        if pr["precision"] == 0 or pr["recall"] == 0:
                            f1 = 0.0
                        else:
                            prec = pr["precision"]
                            rec = pr["recall"]
                            assert isinstance(prec, float) and isinstance(rec, float)
                            f1 = 2 * prec * rec / (prec + rec)
                    except ValueError as e:
                        # print(f"Fail.\n - Category: {category}\n - Error: {e}\n - Gold ranges: {gold_ranges}\n - Pred ranges: {pred_ranges}")
                        print("e", end="")
                        continue

                if category not in f1s_by_category:
                    f1s_by_category[category] = []
                f1s_by_category[category].append(f1)

    return f1s_by_category


def try_parse(data: str) -> int | None:
    try:
        return int(data)
    except:
        return None


def interactive(first_path: str, golden_path: str, second_path: str | None):
    examples = get_examples(first_path, golden_path, second_path)
    n_examples = len(examples)

    idx = 0
    dir = 1
    filter_first: bool | None = None
    filter_second: bool | None = None
    filter = "XX" if examples[0].has_secnd else "X"
    categories = set(e.category for e in examples) | {"all"}
    cat = "all"

    def find_next(curr: int):
        curr += dir
        while curr >= 0 and curr < n_examples:
            curr_example = examples[curr]

            if filter_first is not None:
                if filter_first and not curr_example.first_correct or not filter_first and curr_example.first_correct:
                    curr += dir
                    continue
            if filter_second is not None:
                if filter_second and not curr_example.secnd_correct or not filter_second and curr_example.secnd_correct:
                    curr += dir
                    continue

            if cat != "" and cat.lower() != "all":
                if curr_example.category.lower() != cat.lower():
                    curr += dir
                    continue

            return curr
        return -2

    while True:
        if idx < 0 and idx != -2:
            idx = 0
            dir = 1
        if idx >= n_examples:
            idx = n_examples - 1
            dir = -1

        if idx != -2:
            examples[idx].show()

        print(idx, end=" ")
        if filter != "":
            print(filter, end=" ")
        if cat != "all":
            print(f"[blue]{cat}[reset]", end=" ")
        print()
        line = input().strip()
        line_int = try_parse(line)

        if line == "?":
            print("""
[bold][yellow]?[reset] = show help
[bold][yellow]<empty and press enter>[reset] = next example
[bold][yellow]<number>[reset] = go to example number (ignores filters)
[bold][yellow]filter[reset] <TT/TF/FT/FF/T/F/XX/XT/TX/...> = set a filter for examples (T = must be true, F = must be false, X = any)
[bold][yellow]cat[reset] <category_name or blank> = only show examples from this category (keep blank to show available categories)
[bold][yellow]first[reset] = go to first example matching criteria
[bold][yellow]last[reset] = go to last example matching criteria and invert search direction
[bold][yellow]c1[reset] = copy the first conversation into clipboard
[bold][yellow]c2[reset] = copy the second conversation into clipboard
                 """.strip())
            # TODO: Add "cp first", "cp second", "cp 1", "cp 2", and "cp" commands to copy the thinking traces. (replace "\[" back with "[")
            line = input()

        if line == "":
            idx = find_next(idx)
        elif line_int is not None:
            idx = line_int
        elif line.startswith("filter "):
            rest = line[len("filter "):].strip()
            new_filter = rest[:2]

            if new_filter[0] == "T":
                filter_first = True
            elif new_filter[0] == "F":
                filter_first = False
            else:
                filter_first = None
                new_filter = "X" + new_filter[1:]

            if examples[0].has_secnd:
                if new_filter[1] == "T":
                    filter_second = True
                elif new_filter[1] == "F":
                    filter_second = False
                else:
                    filter_second = None
                    new_filter = new_filter[0] + "X"

            filter = new_filter

            idx = find_next(idx)
        elif line.startswith("cat"):
            rest = line[len("cat"):].strip()
            cat = rest.strip()
            while cat == "":
                print(f"Categories: {', '.join(categories)}")
                print("Enter one of the categories", end=": ")
                cat = input()
                if cat not in categories:
                    cat = ""
        elif line == "last":
            dir = -1
            idx = find_next(n_examples)
        elif line == "first":
            dir = 1
            idx = find_next(-1)
        elif line == "c1":
            if idx >= 0:
                pyperclip.copy(examples[idx].first_conversation)
        elif line == "c2":
            if idx >= 0:
                pyperclip.copy(examples[idx].secnd_conversation)
        else:
            print("unknown command")


def make_csv(args: argparse.Namespace):
    exp_dir = Path(__file__) / "experiments"

    raise NotImplementedError("Not implemented yet (I probably won't make this at all).")

    for file in exp_dir.glob("*.jsonl"):
        ds_name, ds_split, tse_golden = infer_dataset(file)
        correct, correct_cat, detect_cat = get_where_correct(args.answers_file, tse_golden, show_keys_only=args.keys)

    # correct, correct_cat, detect_cat = get_where_correct(args.answers_file, tse_golden, show_keys_only=args.keys)


def main(args: argparse.Namespace):
    assert args.make_csv or args.answers_file, "Call this script with the first argument being the path to the experiment results, or with `--make_csv`."

    if args.make_csv:
        make_csv(args)
        return

    if args.json:
        global print
        # Cancel print.
        def print(*args, **kwargs):
            pass

    # Check we have the right number of output filters (if applied)
    if args.output_filter is not None:
        if args.versus is None:
            assert len(args.output_filter) == 1
        else:
            assert len(args.output_filter) == 2

    ds_name, ds_split, tse_golden = infer_dataset(args.answers_file)

    # Show the inputs
    table = Table(title="Inputs")
    table.add_column("Input")
    table.add_column("Campaign id")
    table.add_column("Strategy")
    table.add_column("Model")

    first_name_infos = infer_name_infos(args.answers_file)
    table.add_row("First", *first_name_infos)
    if args.versus is not None:
        table.add_row("Second", *infer_name_infos(args.versus))
    print(table)

    # Handle anomalies separately
    if "anomaly" in ds_name:
        f1s_cat = get_anomaly_scores(args.answers_file, tse_golden, show_keys_only=args.keys)
        table = Table(title=f"Results on {ds_name}/{ds_split}")
        table.add_column("Category")
        table.add_column("F1")
        for cat, f1s in f1s_cat.items():
            f1_mean = np.mean(np.array(f1s))
            table.add_row(cat, f"{f1_mean:.2f}")
        table.add_section()
        all_f1s = list(chain.from_iterable(f1s_cat.values()))
        all_f1_mean = np.mean(np.array(all_f1s))
        table.add_row("Total", f"{all_f1_mean:.2f}")
        print(table)
        exit()

    # Read datasets
    correct, correct_cat, detect_cat = get_where_correct(args.answers_file, tse_golden, show_keys_only=args.keys)
    correct_cat_vs, detect_cat_vs = None, None
    if args.versus is not None:
        ds_name_vs, ds_split_vs, tse_golden_vs = infer_dataset(args.versus)
        assert ds_name_vs == ds_name and ds_split_vs == ds_split and tse_golden_vs == tse_golden, "The dataset for `--versus` is different!"
        _, correct_cat_vs, detect_cat_vs = get_where_correct(args.versus, tse_golden, max=len(correct))

    # Show the main table
    def get_f1(detect_pairs: Iterable):
        tp = sum(p and t for p, t in detect_pairs)
        p = sum(p for p, t in detect_pairs)
        t = sum(t for p, t in detect_pairs)

        prec = tp / p if p > 0 else 0
        rec = tp / t if t > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        return f1, prec, rec

    def get_acc(corrects: Iterable):
        n_total = sum(1 for _ in corrects)
        n_correct = sum(corrects)
        p_correct = n_correct / n_total
        return p_correct, n_correct, n_total

    show_f1 = ds_name == "time-series-feature-understanding"
    table = Table(title=f"Results on {ds_name}/{ds_split}")
    table.add_column("Category")
    table.add_column("Correct")
    table.add_column("Accuracy")

    if show_f1:
        table.add_column("Detection F1")

    def add_scores(name, p_correct, n_correct, n_total, f1, vs_p_correct = None, vs_n_correct = None, vs_n_total = None, vs_f1 = None):
        exact_fmt = f"{n_correct}/{n_total}"
        acc_fmt = f"{p_correct * 100:.1f}%"
        f1_fmt = f"{f1:.2f}" if f1 else "N/A"

        if vs_n_correct is not None and vs_n_total is not None:
            exact_fmt += f" [grey46](vs {vs_n_correct}/{vs_n_total})"
        if vs_p_correct is not None:
            acc_fmt += f" [grey46](vs {vs_p_correct * 100:.1f}%)"
        if vs_f1 is not None:
            f1_fmt += f" [grey46](vs {vs_f1:.2f})"

        if show_f1:
            table.add_row(name, exact_fmt, acc_fmt,f1_fmt)
        else:
            table.add_row(name, exact_fmt, acc_fmt)

    # campaign strat model
    first_campaign, first_strat, first_model = first_name_infos
    strat = "hybrid" if "hybrid" in first_campaign else \
            "direct" if "direct" in first_campaign else \
            "coder" if "coder" in first_campaign else \
            None
    assert strat is not None
    json_dict = {
        "Strategy": strat, 
        "Raw Data": strat in ["hybrid", "direct"],
        "Can Code": strat in ["hybrid", "coder"],
        "Model": first_model.replace("openai/", ""),
        "Dataset": "TSE" if "exam" in ds_name else "TSFU",
        "Total Accuracy": None # Init here to put it in this position.
    }
    for cat in correct_cat.keys():
        detect_curr_cat = detect_cat[cat]  # list of (pred, true)
        correct_curr_cat = correct_cat[cat]  # list of "correct answer?"

        p_correct, n_correct, n_total = get_acc(correct_curr_cat)
        f1, _, _ = get_f1(detect_curr_cat)

        vs_p_correct, vs_f1 = None, None
        if args.versus is not None:
            assert correct_cat_vs is not None and detect_cat_vs is not None
            correct_vs = correct_cat_vs[cat]  # Gives the same results -> the impl should be correct
            detect_vs = detect_cat_vs[cat]  # Gives the same results -> the impl should be correct
            vs_p_correct, _, _ = get_acc(correct_vs)
            vs_f1, _, _ = get_f1(detect_vs)

        add_scores(cat_name(cat), p_correct, n_correct, n_total, f1, vs_p_correct, None, None, vs_f1)

        json_dict[cat_name(cat)] = p_correct

    # correct = list(chain.from_iterable(correct_cat.values()))  # Gives the same results -> the impl should be correct
    detect = list(chain.from_iterable(detect_cat.values()))  # Gives the same results -> the impl should be correct
    p_correct, n_correct, n_total = get_acc(correct)
    f1, _, _ = get_f1(detect)

    exp_correct = get_expected_correct(str(tse_golden), len(correct))
    exp_correct_sum = sum(exp_correct)

    vs_p_correct, vs_n_correct, vs_n_total, vs_f1 = None, None, None, None
    if args.versus is not None:
        assert correct_cat_vs is not None and detect_cat_vs is not None
        correct_vs = list(chain.from_iterable(correct_cat_vs.values()))  # Gives the same results -> the impl should be correct
        detect_vs = list(chain.from_iterable(detect_cat_vs.values()))      # Gives the same results -> the impl should be correct
        vs_p_correct, vs_n_correct, vs_n_total = get_acc(correct_vs)
        vs_f1, _, _ = get_f1(detect_vs)

    table.add_section()
    add_scores("Expected if random guess", exp_correct_sum / n_total, round(exp_correct_sum, 1), n_total, None)
    table.add_section()
    add_scores("Total", p_correct, n_correct, n_total, f1, vs_p_correct, None, None, vs_f1)

    json_dict["Total Accuracy"] = p_correct
    builtins.print(json.dumps(json_dict))

    # add_scores("Versus", vs_p_correct, vs_n_correct, vs_n_total, vs_f1)

    # if args.json:
        # acc = table.columns[-1]._cells[-1]
        # builtins.print("\n".join(map(str, table.columns)))
        # builtins.print([r for r in table.rows])
        # builtins.print(dict(table))
        # builtins.print("hi")
    print(table)

    # Show the error matrix
    if args.versus is not None:
        correct_2, _, _ = get_where_correct(args.versus, tse_golden, max=len(correct))

        if args.binary:
            print(Rule("correct/wrong"))
            for corr in [correct, correct_2]:
                ones_and_zeros = map(lambda x: "[green]1[reset]" if x else "[red]0[reset]", corr)
                print("".join(ones_and_zeros))

        c_0_0 = sum((not a and not b) for a, b in zip(correct, correct_2))
        c_0_1 = sum((not a and b) for a, b in zip(correct, correct_2))
        c_1_0 = sum((a and not b) for a, b in zip(correct, correct_2))
        c_1_1 = sum((a and b) for a, b in zip(correct, correct_2))

        table = Table(title="Answer matrix")
        table.add_column("First \\ Second")
        table.add_column("Correct")
        table.add_column("Incorrect")
        table.add_row("[bold]Correct[reset]", f"{c_1_1}", f"{c_1_0}")
        table.add_section()
        table.add_row("[bold]Incorrect[reset]", f"{c_0_1}", f"{c_0_0}")
        print(table)

    if args.output_file is not None:
        print(f"Saving the comparison to '{args.output_file}'...")
        save_comparison(args.answers_file, tse_golden, args.versus, correct, correct_2, args.output_file, args.output_filter)

    if args.interactive:
        interactive(args.answers_file, tse_golden, args.versus)


def update_path(args, arg_name):
    orig_val = getattr(args, arg_name, None)
    if orig_val is None:
        return args

    path = Path(getattr(args, arg_name))
    if path.is_dir():
        if (path / "files").exists():
            path = path / "files"

        if (path / "combined.jsonl").exists():
            path = path / "combined.jsonl"
        else:
            files = list(path.glob("*.jsonl"))
            if len(files) == 1:
                path = files[0]
            else:
                print(f"Too many options in '{path}'.")
                for f in files:
                    print(f" - '{str(f)}'")
                raise ArgumentError("Too many options in '{path}'.")

    setattr(args, arg_name, path)
    if str(orig_val) != str(path):
        print(f"updating '{orig_val}' -> '{path}'")
    return args


if __name__ == "__main__":
    args = parser.parse_args()
    for arg_name in ["answers_file", "versus"]:
        args = update_path(args, arg_name)

    main(args)
