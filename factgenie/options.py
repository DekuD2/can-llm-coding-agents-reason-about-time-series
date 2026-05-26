#!/usr/bin/env python3

from typing import Any, Dict, List, Optional, Type

from pydantic import AliasChoices, BaseModel, Field


class QuestionAnsweringOption(BaseModel):
    # Reference: this is what the config for options looks like.
    # `[{'label': 'answer', 'values': ['a', 'b', 'c', 'd']}]`
    label: str = Field(description="The label of the question.")
    # reason: str | None = Field(description="The reason for the answer")
    value: str = Field(description="The selected value")
    # "index"... also exists. It corresponds with the index of the answer.


class QuestionAnsweringModel(BaseModel):
    options: List[QuestionAnsweringOption] = Field(description="The list of choices.")
