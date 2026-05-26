#!/usr/bin/env python3

import heapq
import logging
from litellm.constants import INITIAL_RETRY_DELAY
import numpy as np
import re

from pandas.core.base import extract_array

from factgenie.annotations import SpanAnnotation
from factgenie.prompting.model_apis import ModelAPI
from factgenie.prompting import transforms as t

logger = logging.getLogger("factgenie")

# TODO: DELETE
from time import perf_counter
from rich import print
from rich.rule import Rule

# class FactSplit(t.Transform):
#     def __init__(self, input_field: str, output_field: str):
#         self.output_field = output_field
#         self.input_field = input_field

#     @property
#     def requires_fields(self) -> list[str]:
#         return [self.input_field]

#     @property
#     def outputs_fields(self) -> list[str]:
#         return [self.output_field]

#     # I also tried this text splitter (https://github.com/mediacloud/sentence-splitter). It can properly recognize sentences. It needs the next sentence to either start with a capital letter or a 4-digit number (year). Unfortunately it has problems with markdown, which is a common output of LLMs.
#     @classmethod
#     def iter_sentences_old(cls, text: str):
#         # This regex:
#         #  - '.' and a negative lookahead
#         #    - Can't be followed by another numer, comma, colon, or spaces* lowercase.
#         #      This is needed for decimals (3.5) and abbreviations (e.g. this).
#         #    - Can be followed by an optional \".
#         #  - '?' or '!' followed by an optional \".
#         #  - Extra chunking shouldn't hurt once I show it preceding context.
#         punc_regex = '\\.(?![0-9]|,|:|\\s*[a-z])"?|\\?"?|!"?'
#         parts = [part for part in re.split(punc_regex, text) if len(part) > 2]
#         for part in parts:
#             yield part

#     def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
#         # Go over ever sentence and join each sentence with the rest of the input dictionary.
#         return [{**c, self.output_field: part} for c in current for part in iter_sentences(c[self.input_field])]


# @njit
def extract_fact_tags(fact_tagged_text: str) -> tuple[str, list[int]]:
    fact_re = "</?fact>"
    delimiter_positions = []

    text_resolved_len = 0
    text_that_will_be_deleted_len = 0

    while True:
        match = re.search(fact_re, fact_tagged_text[text_resolved_len:])

        # No more matches.
        if match is None:
            if len(delimiter_positions) % 2 == 1:
                raise ValueError("Odd number of facts.")
            else:
                break

        fact_str = match.group(0)

        # Check for correct alternating order.
        is_fact_opening = fact_str == "<fact>"
        should_be_opening = len(delimiter_positions) % 2 == 0

        if is_fact_opening != should_be_opening:
            raise ValueError(
                "The tags are not matched correctly. They need to go in the order of: <fact>, </fact>, <fact>, </fact>, ..."
            )

        # Add the found tag to the list, accounting in for all the previous tag text that will be deleted.
        match_start = match.start() + text_resolved_len
        match_end = match.end() + text_resolved_len
        delimiter_positions.append(match_start - text_that_will_be_deleted_len)

        # Account for this delimiter for the future iterations.
        text_that_will_be_deleted_len += len(fact_str)
        text_resolved_len = match_end

    text_without_tags = fact_tagged_text.replace("<fact>", "").replace("</fact>", "")
    return text_without_tags, delimiter_positions


def debug_delimit_facts(text_without_tags: str, delimiter_positions: list[int]):
    """
    Adds a '!' on all the `delimiter_positions` to make the correctness easily checkable.
    """
    for p in reversed(delimiter_positions):
        text_without_tags = text_without_tags[:p] + "!" + text_without_tags[p:]
    return text_without_tags


# Without optimization, this function can take 10 seconds.
def calculate_cost_table_full(original_text: str, fact_tagged_text: str, enable_edit_op: bool) -> np.ndarray:
    # In this dynamic programming algorithm, we use a 2D array `cost_table`.
    # The [i, j] index of this table states the edit cost:
    #  • to match the first *i* characters of the `original_text`,
    #  • using up the first *j* characters of the `fact_tagged_text`.
    # NOTE: indices 0 mean "no characters matched"/"no characters used".
    cost_table = np.full([len(original_text) + 1, len(fact_tagged_text) + 1], -1, dtype=np.int32)

    # We initialize to top and left border to simplify the algorithm.
    cost_table[0, :] = np.arange(cost_table.shape[1])
    cost_table[:, 0] = np.arange(cost_table.shape[0])

    for i in range(1, cost_table.shape[0]):
        ti = i - 1  # i-th letter index in `original_text`.
        for j in range(1, cost_table.shape[1]):
            tj = j - 1  # j-th letter index in `fact_tagged_text`.
            if original_text[ti] == fact_tagged_text[tj]:
                # If the new letter in `original_text` matches what was entered in `fact_tagged_text`, then that's the best it can be.
                cost_table[i, j] = cost_table[i - 1, j - 1]
            else:
                insert_cost = cost_table[i - 1, j] + 1  # Insert
                delete_cost = cost_table[i, j - 1] + 1  # Delete
                if enable_edit_op:
                    edit_cost = cost_table[i - 1, j - 1] + 1
                    cost_table[i, j] = min(insert_cost, delete_cost, edit_cost)
                else:
                    cost_table[i, j] = min(insert_cost, delete_cost)

    return cost_table


# To understand `calculate_cost_table_a_star`, look into the simpler `calculate_cost_table_full` first. It's basically the same thing except it calculates only the fastest path in that table using techniques from the A* algorithm.
# It's about ~25 on most cases and takes less than 0.1 seconds when there is at most 8 errors (compared to the `calculate_cost_table_full` taking e.g. 12 seconds).
def calculate_cost_table_a_star(original_text: str, fact_tagged_text: str, enable_edit_op: bool) -> np.ndarray:
    int_max = np.iinfo(np.int32).max

    def new_empty_cost_table(size_i: int, size_j: int) -> np.ndarray:
        # In this dynamic programming algorithm, we use a 2D array `cost_table`.
        # The [i, j] index of this table states the edit cost:
        #  • to match the first *i* characters of the `original_text`,
        #  • using up the first *j* characters of the `fact_tagged_text`.
        # NOTE: indices 0 mean "no characters matched"/"no characters used".
        # This `cost_table` is defined below.
        cost_table = np.full([size_i, size_j], int_max, dtype=np.int32)

        # We initialize to top and left border to simplify the algorithm.
        # This isn't strictly necessary here and might be a bit of waste of time, but will make the algorithm faster (and take only O(n + m) time in C++).
        cost_table[0, :] = np.arange(cost_table.shape[1])
        cost_table[:, 0] = np.arange(cost_table.shape[0])

        return cost_table

    # The heuristic for A* algorithm should basically be the "minimal attainable cost".
    # In this case, it's "going by the diagonal with 0 cost everywhere and then straight right/down (= inserts/deletes)".
    # The second term is to make it prefer furthest elements first. Kind of like a second heuristic. From my experiments I believe this speeds it up by tens of percents.
    def heuristic(i: int, j: int, cost_table: np.ndarray):
        remaining_i = cost_table.shape[0] - i
        remaining_j = cost_table.shape[1] - j
        return abs(remaining_i - remaining_j) + 1 / (remaining_i + remaining_j)

    def add_element_to_fringe(i: int, j: int, fringe: list, cost_table: np.ndarray):
        if i >= cost_table.shape[0] or j >= cost_table.shape[1]:
            # print(f"skipping ({i}, {j})")
            return

        ti = i - 1  # i-th letter index in `original_text`.
        tj = j - 1  # j-th letter index in `fact_tagged_text`.

        if original_text[ti] == fact_tagged_text[tj]:
            cost = cost_table[i - 1, j - 1].item()
        else:
            insert_cost = cost_table[i - 1, j].item() + 1  # Insert
            delete_cost = cost_table[i, j - 1].item() + 1  # Delete
            if enable_edit_op:
                edit_cost = cost_table[i - 1, j - 1].item() + 1
                cost = min(insert_cost, delete_cost, edit_cost)
            else:
                cost = min(insert_cost, delete_cost)

        h = heuristic(i, j, cost_table)
        heapq.heappush(fringe, (cost + h, cost, (i, j)))

    def resolve_fringe_element(fringe: list, cost_table: np.ndarray):
        _, cost, pos = heapq.heappop(fringe)
        i, j = pos

        # It can maybe happen that we have already resolved this element through another path. That path must've been better than this path so we can ignore this path.
        if cost_table[i, j] != int_max:
            return

        cost_table[i, j] = cost
        add_element_to_fringe(i + 1, j, fringe, cost_table)
        add_element_to_fringe(i, j + 1, fringe, cost_table)
        add_element_to_fringe(i + 1, j + 1, fringe, cost_table)

    # The algorithm starts here.
    cost_table = new_empty_cost_table(len(original_text) + 1, len(fact_tagged_text) + 1)
    fringe = []  # Used with heapq for speed. Elements look like: (cost w/ heuristic, cost w/o heuristic, (i, j))

    add_element_to_fringe(1, 1, fringe, cost_table)

    last_i = cost_table.shape[0] - 1
    last_j = cost_table.shape[1] - 1
    while cost_table[last_i, last_j] == int_max:
        # print("fringe length:", len(fringe))
        resolve_fringe_element(fringe, cost_table)

    return cost_table


def identify_facts(original_text: str, fact_tagged_text: str, enable_edit_op: bool = True) -> list[int]:
    """
    Uses algorithm based on minimal edit distance to identify fact span locations in the `fact_tagged_text` as if the tags were in the `original_text` instead, allowing for some LLM errors (typo corrections, removal of formatting, any other changes).

    Args:
        original_text: The original text before annotation.
            e.g. "Crude Oil Prices WTI data spans from **January 1986** to **February 2025** and shows significant volatility over tiem."
        fact_tagged_text: The text annotated by LLM.
            e.g. "Crude Oil Prices WTI data spans from <span>January 1986</span> to <span>February 2025</span> and shows significant volatility over time."
        enable_edit_op: The 'edit' operation in edit distance measuring changes one letter into another. This is especially useful for typos. The following example is one edit: "appke" -> "apple". Setting this to false disallows this operation (leaving only 'insert' and 'delete').
    """
    # print(Rule("fact matching start"))
    print(original_text)
    print("\n\n")
    print(fact_tagged_text)
    print("\n\n")

    # fact_tagged_text, delimiter_positions = extract_fact_tags(fact_tagged_text)
    extract: tuple[str, list[int]] = extract_fact_tags(fact_tagged_text)
    fact_tagged_text, delimiter_positions = extract
    print(fact_tagged_text)

    # To understand `calculate_cost_table_a_star`, look into the simpler `calculate_cost_table_full` first.
    cost_table = calculate_cost_table_a_star(original_text, fact_tagged_text, enable_edit_op)

    CODE_ACCEPT = "ACCEPT"
    CODE_EDIT = "EDIT"
    CODE_INSERT = "INSERT"
    CODE_DELETE = "DELETE"
    CODE_END = "END"

    # Reverse search.
    reversed_path = [(cost_table.shape[0] - 1, cost_table.shape[1] - 1, CODE_END)]
    while True:
        i, j, _ = reversed_path[-1]

        # We hit the border -> there is only one way out.
        if i == 0:
            for j in range(j - 1, -1, -1):
                reversed_path.append((0, j, CODE_DELETE))
            break
        elif j == 0:
            for i in range(i - 1, -1, -1):
                reversed_path.append((i, 0, CODE_INSERT))
            break

        # Otherwise deduce how we got here.
        minimum = np.min(
            [cost_table[i - 1, j - 1], cost_table[i - 1, j], cost_table[i, j - 1]]  # edit or accept  # insert
        )  # delete

        # We always check that the path is equal to minimum and that the "costs" check out.
        is_accept = cost_table[i - 1, j - 1] == minimum and cost_table[i - 1, j - 1] == cost_table[i, j]

        is_edit = cost_table[i - 1, j - 1] == minimum and cost_table[i - 1, j - 1] == cost_table[i, j] - 1

        is_insert = cost_table[i - 1, j] == minimum and cost_table[i - 1, j] == cost_table[i, j] - 1

        is_delete = cost_table[i, j - 1] == minimum and cost_table[i, j - 1] == cost_table[i, j] - 1

        # First try the best step.
        if is_accept:
            reversed_path.append((i - 1, j - 1, CODE_ACCEPT))
        elif enable_edit_op and is_edit:
            reversed_path.append((i - 1, j - 1, CODE_EDIT))
        elif is_insert:
            reversed_path.append((i - 1, j, CODE_INSERT))
        elif is_delete:
            reversed_path.append((i, j - 1, CODE_DELETE))
        else:
            raise Exception("ALGORITHM ERROR (`def identify_facts(...)`)")

    print("Reverse search finished.")
    path = list(reversed(reversed_path))

    adjusted_delimiter_positions = []
    current_delimiter_shift = 0

    # In path reconstruction, when deleting or inserting, modify the relevant elements.
    # Lastly, extract fact start-end tuples and return them.

    for to_point in path:
        # Once we fixed all the delimiter positions, we are finished.
        if len(delimiter_positions) == 0:
            break

        # Deconstruct the step into where we ended up and the step we took to get there.
        # Because of recording steps in a reversed fashion, starting form the end, we always put the correct operation one step too late (which reversed becomes one step too early).
        to_i, to_j, _ = to_point

        # Usually we would just check for `to_j == delimiter_positions[0]`.
        # However, this causes inconsistent behavior when there are inserts after the closing fact-tag. E.g. "**fact**" will get captured as "!**fact!**".
        # Option 1:
        #  • `(len(delimiter_positions) % 2)` --> "!**fact**!"
        #  • Breaks with "**fact:**" as the ending point is just before ':'.
        # Option 2:
        #  • `(len(delimiter_positions) % 2 == 1)` --> "**!fact!**"
        #  • Seems to work better.
        # (The '!' syntax is show the fact locations.)

        while len(delimiter_positions) > 0 and to_j == delimiter_positions[0] + (len(delimiter_positions) % 2 == 0):
            current_delimiter_shift = to_i - to_j
            adjusted_delimiter_positions.append(delimiter_positions[0] + current_delimiter_shift)
            delimiter_positions = delimiter_positions[1:]

    ### Not needed after switching to 'Option 2'.
    # # Because of the `+ (len(delimiter_positions) % 2)`, if there is a closing tag at the absolute end of the text, it won't get processed. Here we fix it.
    # if len(delimiter_positions) == 1:
    #     adjusted_delimiter_positions.append(delimiter_positions[0] + current_delimiter_shift)
    #     delimiter_positions = delimiter_positions[1:]

    # print(f"len(delimiter_positions) = {len(delimiter_positions)}")
    print(debug_delimit_facts(original_text, adjusted_delimiter_positions))
    # print(Rule("fact matching end"))
    assert len(delimiter_positions) == 0
    return adjusted_delimiter_positions


from factgenie.prompting import transforms as t


class DebugShowFacts(t.Transform):
    def __init__(
        self,
        input_field: str,
        output_field: str,
    ):
        self.input_field = input_field
        self.output_field = output_field

    @property
    def requires_fields(self) -> list[str]:
        return [self.input_field, "text"]

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    def parse_annotations(self, c: dict, api: ModelAPI):
        """
        Parse annotations from JSON and validate them.

        Args:
            text: The text to be annotated.
            annotations_json: A JSON string containing the annotations.

        Returns:
            A list of validated annotations.
        """

        text = c["text"]
        text_with_facts = c[self.input_field]

        positions = identify_facts(text, text_with_facts)

        annotation_list = []
        for p_from, p_to in zip(positions[::2], positions[1::2]):
            span = SpanAnnotation(reason="debug", text=text[p_from:p_to], annotation_type=0)
            span = span.model_dump()
            span["start"] = p_from
            span["end"] = p_to
            span["type"] = 0
            annotation_list.append(span)

        return annotation_list

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return t.derive_field(current, api, self.parse_annotations, self.output_field)
