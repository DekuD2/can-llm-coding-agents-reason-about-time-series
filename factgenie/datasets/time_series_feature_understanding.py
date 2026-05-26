import json
import logging
import os
import pandas as pd
import pandas as pd
import plotly.express as px

logger = logging.getLogger("factgenie")

from collections import OrderedDict
from io import StringIO
from pathlib import Path
from time import perf_counter

from factgenie.datasets.dataset import Dataset
from factgenie.datasets.time_series_exam import deep_round


TAG = "tag"
TYPE = "type"
CORRELATION_TYPE = "correlation_type"
ADD_LAGGED_CORRELATION = "add_lagged_correlation"
CSV_PATH = "csv_file"

QUESTION = "question"
OPTIONS = "options"
TS = "ts" 
TS_KEYS = "ts_keys"
TS_KEYS_TIME = "ts_keys_time"
TS_KEYS_OTHER = "ts_keys_other"
ANSWER = "answer"
TS_FORMATTED = "ts_formatted"


def process(item: dict, dir: Path):
    cat = item["category"]

    # Created from https://aclanthology.org/2024.emnlp-main.1204.pdf page 30+
    # Since originally it is split into detection + classification, I need to consider the option of "none" in most questions. I added it in such a way to match the phrasing.
    if cat == "trend":
        item[QUESTION] = "Select one of the following answers:"
        answer_map = OrderedDict({
            "up": "The time series has a positive trend",
            "down": "The time series has a negative trend",
            "none": "The time series has no trend",
        })
        item[ANSWER] = answer_map[item[TAG]]
        item[OPTIONS] = list(answer_map.values())

    elif cat == "seasonality":
        item[QUESTION] = """
Given the following definitions:
 - Fixed-period: Regular, predictable seasonal patterns occurring at fixed intervals (e.g., daily, weekly, monthly).
 - Shifting Period: Seasonal patterns where the length of the period shifts over time.
 - Multiple seasonality: Presence of multiple overlapping seasonal patterns (e.g., both weekly and monthly seasonality)

Select one of the available answers:
"""
        answer_map = OrderedDict({
            "fixed": "The time series has fixed-period seasonality",
            "shift_patt": "The time series has a shift in seasonal pattern",
            "multiple": "The time series has multiple seasonal patterns",
            "none": "The time series does not exhibit any of these seasonal patterns",
        })
        item[ANSWER] = answer_map[item[TYPE]]
        item[OPTIONS] = list(answer_map.values())

    elif cat == "outliers":
        item[QUESTION] = """
Given the following definitions:
 - Spike: a sudden and brief deviation from the overall pattern of the data.
 - Level shift: a sudden and lasting change in the average value of the series.
 - Temporal disruption: an interval where data is missing or not recorded.

Select one of the following answers that best describes the provided time series:
        """
        answer_map = OrderedDict({
            "spike": "The time series has one or more spikes",
            "level_shift": "The time series has a level shift",
            "temporal_disruption": "The time series has a temporal disruption",
            "none": "The time series exhibits none of these anomalies",  # neither or no irregularities?
        })
        item[ANSWER] = answer_map[item[TYPE]]
        item[OPTIONS] = list(answer_map.values())
       
    elif cat == "volatility":
        item[QUESTION] = """
Given the following definitions:
 - Constant Volatility: The degree of variation in the time series remains consistent and predictable over time.
 - Trending Volatility: The level of variation in the time series shows a clear increasing or decreasing trend over time.
 - Clustered Volatility: The time series exhibits periods where volatility is significantly higher or lower, with these periods tending to cluster together.
 - Dynamic Volatility: The volatility of the time series changes over time in response to external factors (e.g., leverage effect where the volatility of the time series tends to increase when the series experiences negative returns).

Select one of the following answers:
        """
        answer_map = OrderedDict({
            "constant": "The time series has constant volatility",
            "trending": "The time series has trending volatility",
            "clustered": "The time series has clustered volatility",
            "leverage": "The time series has dynamic volatility",
            "none": "The time series does not exhibit any of these volatility patterns",
        })
        item[ANSWER] = answer_map[item[TYPE]]
        item[OPTIONS] = list(answer_map.values())

    elif cat == "structural_break":
        item[QUESTION] = """
Given the following definitions:
 - Regime Change: A shift in the time series data’s statistical properties, such as mean, variance, or auto-correlation, that persists over time. This change is often gradual and represents a new phase or ’regime’ in the data.
 - Structural Break: An abrupt change in the time series data that leads to a new level or trend. This change is typically sudden and can be linked to specific events or shifts in the underlying process.

Examine the provided time series data and select the correct option:
        """
        answer_map = OrderedDict({
            "regime_shift": "The time series data exhibits a Regime Change",
            "parameter_shift": "The time series data exhibits a Structural Break",
            "no_structural_break": "The time series data exhibits neither a Regime Change nor a Structural Break",
        })
        item[ANSWER] = answer_map[item[TYPE]]
        item[OPTIONS] = list(answer_map.values())

    elif cat == "statistical_property":
        item[QUESTION] = """
Considering the data provided, does the time series exhibit fat tails? Fat tails refer to a higher likelihood of extreme values compared to a normal distribution, indicating a higher probability of observing significant positive or
negative deviations.
        """
        answer_map = OrderedDict({
            "fat_tail": "Yes",
            "no_fat_tail": "No",
        })
        item[ANSWER] = answer_map[item[TYPE]]
        item[OPTIONS] = list(answer_map.values())

    elif cat == "stationarity":
        item[QUESTION] = """
Given the following definitions of non-stationary types in time series data:
  - Trend Change: The time series exhibits a significant shift in its underlying trend, indicating a change in the mean over time.
  - Variance Change: The time series shows a change in its variability or spread.
  - Seasonality: The time series displays regular and predictable patterns that repeat over a certain period.
  - Trend and Seasonality: The time series exhibits both a significant underlying trend and seasonal patterns. This type combines elements of both trend changes and predictable seasonal fluctuations.

Select one of the following answers based on your analysis of the time series:
        """
        answer_map = OrderedDict({
            "trend_change": "The time series has a trend change",
            "variance_change": "The time series has a variance change",
            "seasonality": "The time series has seasonality",
            "trend_seasonality": "The time series has both trend and seasonality",
            "stationary": "The time series is stationary",
        })
        item[ANSWER] = answer_map[item[TAG]]
        item[OPTIONS] = list(answer_map.values())

    elif cat == "correlation":
        item[QUESTION] = """
Select one of the following answers:
        """
        answer_map = OrderedDict({
            "positive": "The time series are positively correlated",
            "negative": "The time series are negatively correlated",
            "zero": "The time series are not correlated",
        })
        item[ANSWER] = answer_map[item[CORRELATION_TYPE]]
        item[OPTIONS] = list(answer_map.values())

    elif cat == "lagged_correlation":
        item[QUESTION] = """
Given the following definitions:
 - Direct Correlation: The two time series show a direct, immediate relationship between their values, where changes in one series directly influence the other in a straightforward manner.
 - Direct Lagged Correlation: The two time series demonstrate a delayed relationship, where changes in one series influence the other after a certain lag period.
 - Inverse Correlation: The two time series exhibit an inverse or negative relationship between their values, where an increase in one series typically leads to a decrease in the other, and vice versa.
 - Inverse Lagged Correlation: The two time series show a relationship where changes in one series negatively influence the other after a certain lag period, suggesting that past increases in one series lead to future decreases in the other, and vice versa.

Select one of the following answers that best describes the relationship between the two time series:
"""
        answer_map = OrderedDict({
            (False, "positive"): "The two time series exhibit direct correlation",
            (False, "negative"): "The two time series exhibit inverse correlation",
            (True, "positive"): "The two time series exhibit direct lagged correlation",
            (True, "negative"): "The two time series exhibit inverse lagged correlation.",
            (True, "zero"): "The two time series exhibit no correlation",
            (False, "zero"): "The two time series exhibit no correlation",
        })
        item[ANSWER] = answer_map[(bool(item[ADD_LAGGED_CORRELATION]), item[CORRELATION_TYPE])]
        item[OPTIONS] = list(answer_map.values())[:-1]

    # elif cat == "TEMPLATE":
    #     item[QUESTION] = """
    #     """
    #     answer_map = ({
            
    #     })
    #     item[ANSWER] = answer_map[item[???]]
    #     item[OPTIONS] = answer_map.values()
    else:
        raise KeyError(f"Unknown category {cat}!")

    item[QUESTION] = item[QUESTION].strip()
    df = pd.read_csv(dir / item[CSV_PATH])

    old = df.columns
    assert len(old) in [2, 3], f"Unexpected number of columns ({len(old)})"
    new = ["time", "value"] if len(old) == 2 else ["time", "value1", "value2"]
    df = df.rename(columns={o: n for (o, n) in zip(old, new)})
    item[TS] = df.to_json()
    new = list(map(lambda x: f"'{x}'", new))
    item[TS_KEYS] = ", ".join(new)
    item[TS_KEYS_TIME] = new[0]
    item[TS_KEYS_OTHER] = ", ".join(new[1:])

    item["answer_index"] = item["options"].index(item["answer"])
    item["options"] = [f"{chr(ord('A') + i)}) {option}" for i, option in enumerate(item["options"])]
    item["A)options"] = "\n".join(" " + option for option in item["options"])

    item[TS_FORMATTED] = "\n".join(
        ", ".join(f"{col}: {row[col]}" for col in df.columns)
        for _, row in df.iterrows()
    )

    return item

class TimeSeriesFeatureUnderstanding(Dataset):
    SIXTY_INDICES = [22, 67, 107, 164, 170, 181, 190, 223, 238, 379, 399, 427, 476, 494, 516, 559, 625, 627, 629, 630, 666, 671, 760, 765, 909, 919, 922, 959, 991, 1041, 1114, 1129, 1170, 1267, 1291, 1293, 1308, 1352, 1374, 1376, 1389, 1420, 1479, 1481, 1493, 1495, 1521, 1537, 1541, 1557, 1588, 1653, 1694, 1697, 1762, 1819, 1836, 1876, 1912, 1937]

    def __init__(self, *vargs, **kwargs):
        self.example_shown = False
        super().__init__(*vargs, **kwargs)

    def load_examples(self, split, data_path):
        verbose = not self.example_shown

        sixty_ds = split == "sixty"
        if sixty_ds:
            input = "test"
        else:
            input = split

        self.example_shown = True

        # Inter-file key intersection:
        #   'idx', 'start_date', 'end_date', 'description_qualitative', 'plot_file', 'type', 'csv_file', 'description', 'category'
        #   'category': constant within file
        #   'type': the answer
        #   'description': contains the answer in words
        #   'description_qualitative': contains the answer in words

        # Inter-file key (union - intersection):
        #   'sudden_spike_count', 'tag', 'sudden_spikes_loc', 'parameter_shift_type', 'correlation_coefficient', 'max_date_Series_1', 'correlation_change', 'random_date', 'correlation_choice', 'max_date', 'correlation_type', 'regime_shift_type', 'high_peaks', 'peaks_dates', 'sudden_spikes_val', 'min_date_Series_2', 'yesno', 'drop_date_Series_2', 'correlated_series', 'season_period', 'value_on_date', 'min_value_Series_1', 'step_spike_val', 'num_time_series', 'value_on_date_Series_1', 'step_spike_start', 'random_date_Series_1', 'min_value', 'distribution_change_type', 'min_date', 'lag', 'level_shift_val', 'avg_volatility', 'max_date_Series_2', 'drop_date_Series_1', 'low_peaks', 'add_lagged_correlation', 'level_shift_loc', 'max_value', 'temporal_disruption_loc_end', 'random_date_Series_2', 'low_peaks_dates', 'min_date_Series_1', 'step_spike_duration', 'min_value_Series_2', 'max_value_Series_1', 'largest_drop_Series_2', 'largest_drop_Series_1', 'frequency', 'shift_point', 'degree_of_freedom_fat_tail', 'max_value_Series_2', 'temporal_disruption_loc_start', 'value_on_date_Series_2'
        examples = []
        answers = []
        start = perf_counter()

        # item["options"] = [f"{chr(ord('A') + i)}) {option}" for i, option in enumerate(item["options"])]
        # item["A)options"] = "\n".join(" " + option for option in item["options"])

        dir = Path(f"{data_path}/{input}")
        jsons = dir.glob("*.jsonl")

        idx = 0
        for json_file in jsons:
            if verbose:
                logger.info(f"Processing file {json_file.name}...")

            with open(json_file, "r") as file:
                lines = file.readlines()
                for line in lines:
                    if sixty_ds and idx not in self.SIXTY_INDICES:
                        idx += 1
                        continue

                    j = json.loads(line)
                    j = process(j, dir)
                    j["id"] = idx

                    examples.append(j)

                    answers.append(
                        {
                            "type": "multiple_choice_question_answering",
                            "example_idx": idx,
                            "quesiton": j["question"],
                            "options": [o[3:] for o in j["options"]],
                            "answer": j["answer"],
                            "answer_index": j["answer_index"],
                            "category": j["category"],
                        }
                    )

                    idx += 1

        answers_file = Path(data_path) / f"{split}-answers.jsonl"
        if not answers_file.exists():
            logger.info(f"Creating answers file for the QA task. File location: '{answers_file}'")

            os.makedirs(data_path, exist_ok=True)
            with open(answers_file, "w") as f:
                f.writelines(json.dumps(a) + "\n" for a in answers)

        # DEBUG PRINT
        if verbose:
            def tree(example, indent=1) -> str:
                if type(example) is dict:
                    text = ""
                    for key in example.keys():
                        text += f"\n{indent * ' '}• {key}"
                        text += tree(example[key], indent + 2)
                    return text
                elif type(example) is list:
                    text = " (list)"
                    text += tree(example[0], indent)
                    return text
                else:
                    return ""

            # logger.info("Time Series Dataset example structure:" + tree(examples[0]))

        return examples

    def render_figs(self, example):
        fig_htmls = []
        df = pd.read_json(StringIO(example[TS]))
        ncols = len(df.columns)
        fig = px.line(
            df,
            x="time",
            y="value" if ncols == 2 else ["value1", "value2"],
            title="time series",
            template="plotly_white",
            # hover_data=["time", *features.keys()]
        )
        fig_htmls.append(fig.to_html(include_plotlyjs="cdn"))

        return fig_htmls

    def render(self, example):
        # fig_htmls = []

        # df = pd.read_json(StringIO(example[TS]))

        # ncols = len(df.columns)

        # fig = px.line(
        #     df,
        #     x="time",
        #     y="value" if ncols == 2 else ["value1", "value2"],
        #     title="time series",
        #     # hover_data=["time", *features.keys()]
        # )
        # fig_htmls.append(fig.to_html(include_plotlyjs="cdn"))

        fig_htmls = self.render_figs(example)

        html = ""

        return (
            """<div id="graph">"""
            + "\n".join(fig_htmls)
            + """</div><div class="root" style="margin-top: 40px">"""
            + html
            + """</div>"""
        )
