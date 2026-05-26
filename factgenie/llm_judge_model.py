#!/usr/bin/env python

import dataclasses
import json
import re
import orjson

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AnnItem:
    name: str
    choices: list
    explanations: list[str]
    eval: list[bool]

    @classmethod
    def from_dict(cls, d: dict):
        return AnnItem(**d)


@dataclass
class AnnCategory:
    name: str
    items: list[AnnItem]

    @classmethod
    def from_dict(cls, d: dict):
        return AnnCategory(name=d["name"], items=[AnnItem.from_dict(i) for i in d["items"]])


class Ann:
    def __init__(self, text_per_model: list[str]):
        self._text_per_model: list[str] | None = text_per_model
        self._ann_categories: list[AnnCategory] | None = None

    @classmethod
    def from_dict(cls, d: dict):
        a = Ann([""])
        a._text_per_model = None
        a._ann_categories = [AnnCategory.from_dict(ac) for ac in d["categories"]]
        return a

    def _get_sections(self, text: str):
        sections = {}
        sections["categories"] = {}
        j = json.loads(text)
        for k, v in j.items():
            if isinstance(v, dict):
                sections[k] = v
            else:
                sections["categories"][k] = v
        return sections

    def _items_from_sections(self, section_multiple_models: list[dict]):
        first = section_multiple_models[0]
        items: list[AnnItem] = []
        for k in first.keys():
            # Skip legacy
            if "other" in k:
                continue

            # Explanations go separately
            if "explanation" in k:
                # Check the item this explanation is to exists as well...
                assert k[: -(len("_explanation"))] in first.keys()
                continue

            k_exp = k + "_explanation"
            assert k_exp in first.keys(), f"{k_exp} not in [{', '.join(first.keys())}]"

            choices = [s[k] for s in section_multiple_models]
            explanations = [s[k_exp] for s in section_multiple_models]
            items.append(AnnItem(name=k, choices=choices, explanations=explanations, eval=[]))

        return items

    def _parse_texts_to_categories(self):
        if self._text_per_model is None:
            return self._ann_categories

        # Translate each into a dictionary like {section1: {sub1: True, sub1_explanation: blablabala}}.
        sections_per_model = [self._get_sections(text) for text in self._text_per_model]

        keys = sections_per_model[0].keys()

        self._text_per_model = None
        self._ann_categories = [
            AnnCategory(name=k, items=self._items_from_sections([s[k] for s in sections_per_model])) for k in keys
        ]
        return self._ann_categories

    @property
    def categories(self) -> list[AnnCategory]:
        if self._ann_categories is None:
            self._parse_texts_to_categories()

        return self._ann_categories


@dataclass
class DatasetItem:
    example_index: int
    conversation: str
    annotations: Ann
    correct: bool | None = None
    category: str | None = None

    @classmethod
    def from_dict(cls, d: dict):
        return DatasetItem(
            example_index=d["example_index"],
            conversation=d["conversation"],
            annotations=Ann.from_dict(d["annotations"]),
        )


# --- The dataset class containing the whole analysis ---
@dataclass
class Evaluator:
    filename: str
    api_provider: str
    model: str
    prompt_strat: str
    prompt_template: str
    system_msg: str
    model_args: dict
    extra_args: dict
    campaign_id: str
    annotator_id: str

    @classmethod
    def from_dict(cls, d: dict):
        return Evaluator(
            filename=d["filename"],
            api_provider=d["api_provider"],
            model=d["model"],
            prompt_strat=d["prompt_strat"],
            prompt_template=d["prompt_template"],
            system_msg=d["system_msg"],
            model_args=d["model_args"],
            extra_args=d["extra_args"],
            campaign_id=d["campaign_id"],
            annotator_id=d["annotator_id"],
        )


@dataclass
class DatasetInfo:
    dataset: str
    split: str

    @classmethod
    def from_dict(cls, d: dict):
        return DatasetInfo(dataset=d["dataset"], split=d["split"])


@dataclass
class DatasetEval:
    info: DatasetInfo
    evaluators: list[Evaluator]
    examples: list[DatasetItem]

    @classmethod
    def from_json(cls, json: str):
        d = orjson.loads(json)
        return DatasetEval(
            info=DatasetInfo.from_dict(d["info"]),
            evaluators=[Evaluator.from_dict(ev) for ev in d["evaluators"]],
            examples=[DatasetItem.from_dict(ex) for ex in d["examples"]],
        )

    def add_correct(self, outputs_path: str, answers_path: str):
        with open(outputs_path, mode="r") as f_o:
            with open(answers_path, mode="r") as f_a:
                o_lines = list(map(json.loads, f_o.readlines()))
                a_lines = list(map(json.loads, f_a.readlines()))
        for ex in self.examples:
            o = o_lines[ex.example_index]
            a = a_lines[ex.example_index]
            # Make sure the indexes match.
            assert(o["example_idx"] == ex.example_index)
            assert(a["example_idx"] == ex.example_index)
            ex.correct = a['answer_index'] == o['options'][0]['index']
            ex.category = a["category"]

    def to_json(self):
        def serialize_default(obj):
            if isinstance(obj, Ann):
                return {"categories": obj.categories}

            raise TypeError(f"Cannot serialize type {type(obj)}")

        return orjson.dumps(self, default=serialize_default)


# Just for initial loading
@dataclass
class FileProxy:
    conversation: str
    annotations: list[str]


TRIM_RAW_JSON = True

def load_file(path: str) -> list[FileProxy]:
    def get_pair(line):
        j = json.loads(line)
        metadata = conv = j["metadata"]
        if "agent_conversation" in metadata.keys():
            conv = metadata["agent_conversation"]
        else:
            conv = metadata["prompt"]

        if TRIM_RAW_JSON:
            conv = re.sub("([Gg]iven this data:\n\n?)```json[^`]*```", "```json\n(...trimmed)\n```", conv)

        # Find the correct answer and append it.
        prompt = metadata["prompt"]
        ca = re.findall("The correct answer was (.*?)$", prompt, re.MULTILINE)
        if len(ca) > 0:
            conv += f"\n=== The correct answer was '{ca[0]}' ==="

        ann = j["output"]  # TODO: use j["metadata"]["agent_conversation"] instead
        return FileProxy(conversation=conv, annotations=[ann])

    with open(path, "r") as f:
        return [get_pair(line) for line in f.readlines()]


def get_metadata(path: str) -> tuple[DatasetInfo, Evaluator]:
    with open(path, "r") as f:
        line = f.readline()
        j = json.loads(line)
        m = j["metadata"]

        ds_info = DatasetInfo(dataset=j["dataset"], split=j["split"])

        evaluator_keys = {f.name for f in dataclasses.fields(Evaluator)}
        evaluator = Evaluator(filename=Path(path).name, **{k: v for k, v in m.items() if k in evaluator_keys})

    return ds_info, evaluator

