#!/usr/bin/env python3

import logging
import os
import warnings
import unittest

from ast import literal_eval

from factgenie.campaign import CampaignMode
from factgenie.prompting.model_apis import (
    ModelAPI,
    OllamaAPI,
    register_model_api,
    unregistered_model_api_tracker,
)
from factgenie.prompting.strategies import (
    PromptingStrategy,
    register_llm_eval,
    register_llm_gen,
    unregistered_prompting_strategy_tracker,
)

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# also disable info logs from litellm
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Proxy").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)

# disable requests logging
logging.getLogger("httpx").setLevel(logging.ERROR)

logger = logging.getLogger("factgenie")

DIR_PATH = os.path.dirname(__file__)
LLM_ANNOTATION_DIR = os.path.join(DIR_PATH, "annotations")
LLM_GENERATION_DIR = os.path.join(DIR_PATH, "outputs")


class ModelFactory:
    """
    Factory for creating model instances based on configuration.
    This class is responsible for parsing the configuration, selecting the appropriate model API, and initializing the prompting strategy.
    """

    @staticmethod
    def get_model_apis():
        # Only warns once about each unregistered subclass.
        unregistered_model_api_tracker.warn_about_unregistered_subclasses()

        # List of available model APIs that the user can select from, stored as `api_provider` in the config.
        return register_model_api.registered_subclasses

    @staticmethod
    def get_prompt_strategies():
        # Only warns once about each unregistered subclass.
        unregistered_prompting_strategy_tracker.warn_about_unregistered_subclasses()

        return {
            CampaignMode.LLM_GEN: register_llm_gen.registered_subclasses,
            CampaignMode.LLM_EVAL: register_llm_eval.registered_subclasses,
        }

    @staticmethod
    def parse_api_provider(config):
        if "type" in config:
            logger.warning(
                "The `type` field is deprecated. Please use `api_provider` instead. This will be removed in a future version."
            )

        # Supporting the deprecated `type` field
        api_provider = config.get("api_provider", config.get("type"))

        # Supporting the deprecated suffixes
        if api_provider.endswith("_metric"):
            api_provider = api_provider[: -len("_metric")]
        elif api_provider.endswith("_gen"):
            api_provider = api_provider[: -len("_gen")]

        return api_provider

    @staticmethod
    def from_config(config, mode):
        api_provider = ModelFactory.parse_api_provider(config)

        prompt_strat = config.get("prompt_strat", "default")
        if "prompt_strat" not in config:
            logger.warning("Prompting strategy was not specified, using 'default'...")

        model_apis = ModelFactory.get_model_apis()
        prompt_strats = ModelFactory.get_prompt_strategies()[mode]

        # ensure the api_type and prompt_strat are valid
        if api_provider not in model_apis:
            raise ValueError(f"Model type {api_provider} is not implemented.")
        if prompt_strat not in prompt_strats:
            raise ValueError(f"Model type {prompt_strat} is not implemented.")

        return Model(config, mode, model_apis[api_provider](config), prompt_strats[prompt_strat](config, mode))


class Model:
    def __init__(self, config: dict, mode: CampaignMode, model_api: ModelAPI, prompt_strat: PromptingStrategy):
        self.config = config
        self.campaign_mode = mode
        self.parse_model_args()
        self.model_api = model_api
        self.prompt_strat = prompt_strat

    def generate_output(self, data, text=None):
        """For backward compatibility with existing code."""
        return self.prompt_strat.get_output(api=self.model_api, data=data, text=text)

    def get_annotator_id(self):
        return "llm-" + ModelFactory.parse_api_provider(self.config) + "-" + self.config["model"]

    def get_config(self):
        return self.config

    def parse_model_args(self):
        if "model_args" not in self.config:
            return

        # implicitly convert all model_args to literals based on their format
        for arg in self.config["model_args"]:
            try:
                self.config["model_args"][arg] = literal_eval(self.config["model_args"][arg])
            except:
                pass

    def validate_config(self, config):
        for field in self.get_required_fields():
            assert field in config, f"Field `{field}` is missing in the config. Keys: {config.keys()}"

        for field, field_type in self.get_required_fields().items():
            assert isinstance(
                config[field], field_type
            ), f"Field `{field}` must be of type {field_type}, got {config[field]=}"

        for field, field_type in self.get_optional_fields().items():
            if field in config:
                assert isinstance(
                    config[field], field_type
                ), f"Field `{field}` must be of type {field_type}, got {config[field]=}"
            else:
                # set the default value for the data type
                config[field] = field_type()

        # warn if there are any extra fields
        for field in config:
            if field not in self.get_required_fields() and field not in self.get_optional_fields():
                logger.warning(f"Field `{field}` is not recognized in the config.")




# Experiments start...
import json
from rich import print


def tool_test(api: ModelAPI, messages: list[dict], prompt_strat_kwargs: dict, include_tools: bool = True):
    def get_current_weather(location, unit="celsius"):
        return json.dumps({"location": location, "temperature": 10 + (10 if "pr" in location.lower() else 0), "unit": unit})

    get_current_weather_tool = {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. Prague, Czechia",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }

    call_code_tool = {
        "type": "function",
        "function": {
            "name": "call_code",
            "description": "Call python interpreter. The function main will be called automatically with the appropriate arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The python code to call.",
                    },
                },
                "required": ["code"]
            }
        }
    }

    tools = {"tools": [get_current_weather_tool, call_code_tool]} if include_tools else {}
    repsonse = api.get_model_response_with_retries(messages, prompt_strat_kwargs | tools)

    return repsonse


class ModelsTests(unittest.TestCase):
    THOUGHT = "thinking"

    def __init__(self, *vargs, **kwargs):
        super().__init__(*vargs, **kwargs)
        self.api = OllamaAPI(config={"model": "gpt-oss:120b",
                                     "api_url": "http://127.0.0.1:11434",
                                     "num_ctx": 128000,
                                     "num_predict": 8000,
                                     "seed": 42,
                                     "temperature": 0,
                                     "reasoning_effort": "high"})

    def test_weather_tool(self):
        messages = []
        messages.append({"role": "user", "content": "What's the weather in prague and tokyo? Use the provided tools."})
        # response = tool_test(self.api, messages, {})
        # print(response)

        self.assertListEqual([True], [True])


    def test_code_tool(self):
        instructions = """You are given this data:
```
time: 2023-01-02, ts1: 5.5, ts2: 9.4
time: 2023-01-03, ts1: 5.9, ts2: 4.7
time: 2023-01-04, ts1: 5.8, ts2: 5.9
time: 2023-01-05, ts1: 1.8, ts2: 4.4
time: 2023-01-06, ts1: 6.0, ts2: 5.1
time: 2023-01-09, ts1: 9.7, ts2: 5.4
time: 2023-01-10, ts1: 9.1, ts2: 3.0
time: 2023-01-11, ts1: 5.3, ts2: 4.6
time: 2023-01-12, ts1: 9.3, ts2: 5.4
time: 2023-01-13, ts1: 2.8, ts2: 7.7
time: 2023-01-16, ts1: 7.3, ts2: 3.0
time: 2023-01-17, ts1: 5.9, ts2: 2.0
time: 2023-01-18, ts1: 9.8, ts2: 0.5
time: 2023-01-19, ts1: 5.9, ts2: 5.5
time: 2023-01-20, ts1: 11.3, ts2: 0.2
time: 2023-01-23, ts1: 6.8, ts2: 3.9
time: 2023-01-24, ts1: 7.8, ts2: 7.7
time: 2023-01-25, ts1: 10.4, ts2: 4.0
time: 2023-01-26, ts1: 10.3, ts2: 1.7
time: 2023-01-27, ts1: 6.8, ts2: 1.1
time: 2023-01-30, ts1: 6.1, ts2: 3.9
time: 2023-01-31, ts1: 3.8, ts2: 9.2
time: 2023-02-01, ts1: 7.7, ts2: 4.7
time: 2023-02-02, ts1: 4.2, ts2: 7.7
time: 2023-02-03, ts1: 5.1, ts2: 6.2
time: 2023-02-06, ts1: 5.4, ts2: 6.9
time: 2023-02-07, ts1: 4.0, ts2: 7.0
time: 2023-02-08, ts1: 5.1, ts2: 5.5
time: 2023-02-09, ts1: 4.9, ts2: 6.3
time: 2023-02-10, ts1: 2.9, ts2: 5.9
time: 2023-02-13, ts1: 5.9, ts2: 8.7
time: 2023-02-14, ts1: 8.5, ts2: 2.0
time: 2023-02-15, ts1: 8.3, ts2: 2.2
time: 2023-02-16, ts1: 6.8, ts2: 2.4
time: 2023-02-17, ts1: 3.6, ts2: 6.5
time: 2023-02-20, ts1: 7.4, ts2: 5.3
time: 2023-02-21, ts1: 9.5, ts2: 4.1
time: 2023-02-22, ts1: 4.4, ts2: 10.2
time: 2023-02-23, ts1: 2.2, ts2: 6.0
time: 2023-02-24, ts1: 8.7, ts2: 0.8
time: 2023-02-27, ts1: 6.5, ts2: 3.8
time: 2023-02-28, ts1: 5.3, ts2: 10.3
time: 2023-03-01, ts1: 6.5, ts2: 8.5
time: 2023-03-02, ts1: 7.1, ts2: 4.0
time: 2023-03-03, ts1: 1.8, ts2: 5.6
time: 2023-03-06, ts1: 10.4, ts2: -0.4
time: 2023-03-07, ts1: 12.1, ts2: 2.9
time: 2023-03-08, ts1: 5.0, ts2: 6.3
time: 2023-03-09, ts1: 9.4, ts2: 4.8
time: 2023-03-10, ts1: 3.3, ts2: 2.3
time: 2023-03-13, ts1: 8.9, ts2: 1.3
time: 2023-03-14, ts1: 4.5, ts2: 4.5
time: 2023-03-15, ts1: 6.6, ts2: -0.0
time: 2023-03-16, ts1: 7.2, ts2: 7.9
time: 2023-03-17, ts1: 6.3, ts2: 4.4
time: 2023-03-20, ts1: 8.9, ts2: 2.0
time: 2023-03-21, ts1: 6.0, ts2: 7.6
time: 2023-03-22, ts1: 4.5, ts2: 5.6
time: 2023-03-23, ts1: 6.1, ts2: 2.4
time: 2023-03-24, ts1: 5.3, ts2: 5.5
```

And a following question:
```
Select one of the following answers:
 A) The time series are positively correlated
 B) The time series are negatively correlated
 C) The time series are not correlated
```

Your goal is to answer this question. You have an access to a python code interpreter tool to gather evidence for your answer. Gather evidence first before answering. When you know the correct answer, output it word-for-word without outputting anything else.

When using the call_code tool, the code should contain a single method `main` taking a single parameter `dict_of_dfs` of type `dict[str, pd.DataFrame]`. The argument `dict_of_dfs` that the function receives is a dictionary containing the following key(s): 'ts1', 'ts2'. Each dataframe contains a single unnamed column. The python version is 3.9. The libraries available are "pandas==2.2.3", "numpy==1.26.4", "scipy==1.14.1", and "statsmodels==0.14.5".

Here is an example code start:

import pandas as pd
import numpy as np

def main(dict_of_dfs: dict[str, pd.DataFrame]):
    # `dict_of_dfs` is a dictionary containing the following key(s): 'ts1', 'ts2'.
    # Under each key is a pandas dataframe with a single unnamed column."""

        messages = []
        messages.append({"role": "user", "content": instructions})
        response = tool_test(self.api, messages, {})
        print(response)
        print("code: \n" + str(response.choices[0].message.tool_calls[0].function.arguments))

        self.assertListEqual([True], [True])


    def test_code_without(self):
        instructions = """You are given this data:
```
time: 2023-01-02, ts1: 5.5, ts2: 9.4
time: 2023-01-03, ts1: 5.9, ts2: 4.7
time: 2023-01-04, ts1: 5.8, ts2: 5.9
time: 2023-01-05, ts1: 1.8, ts2: 4.4
time: 2023-01-06, ts1: 6.0, ts2: 5.1
time: 2023-01-09, ts1: 9.7, ts2: 5.4
time: 2023-01-10, ts1: 9.1, ts2: 3.0
time: 2023-01-11, ts1: 5.3, ts2: 4.6
time: 2023-01-12, ts1: 9.3, ts2: 5.4
time: 2023-01-13, ts1: 2.8, ts2: 7.7
time: 2023-01-16, ts1: 7.3, ts2: 3.0
time: 2023-01-17, ts1: 5.9, ts2: 2.0
time: 2023-01-18, ts1: 9.8, ts2: 0.5
time: 2023-01-19, ts1: 5.9, ts2: 5.5
time: 2023-01-20, ts1: 11.3, ts2: 0.2
time: 2023-01-23, ts1: 6.8, ts2: 3.9
time: 2023-01-24, ts1: 7.8, ts2: 7.7
time: 2023-01-25, ts1: 10.4, ts2: 4.0
time: 2023-01-26, ts1: 10.3, ts2: 1.7
time: 2023-01-27, ts1: 6.8, ts2: 1.1
time: 2023-01-30, ts1: 6.1, ts2: 3.9
time: 2023-01-31, ts1: 3.8, ts2: 9.2
time: 2023-02-01, ts1: 7.7, ts2: 4.7
time: 2023-02-02, ts1: 4.2, ts2: 7.7
time: 2023-02-03, ts1: 5.1, ts2: 6.2
time: 2023-02-06, ts1: 5.4, ts2: 6.9
time: 2023-02-07, ts1: 4.0, ts2: 7.0
time: 2023-02-08, ts1: 5.1, ts2: 5.5
time: 2023-02-09, ts1: 4.9, ts2: 6.3
time: 2023-02-10, ts1: 2.9, ts2: 5.9
time: 2023-02-13, ts1: 5.9, ts2: 8.7
time: 2023-02-14, ts1: 8.5, ts2: 2.0
time: 2023-02-15, ts1: 8.3, ts2: 2.2
time: 2023-02-16, ts1: 6.8, ts2: 2.4
time: 2023-02-17, ts1: 3.6, ts2: 6.5
time: 2023-02-20, ts1: 7.4, ts2: 5.3
time: 2023-02-21, ts1: 9.5, ts2: 4.1
time: 2023-02-22, ts1: 4.4, ts2: 10.2
time: 2023-02-23, ts1: 2.2, ts2: 6.0
time: 2023-02-24, ts1: 8.7, ts2: 0.8
time: 2023-02-27, ts1: 6.5, ts2: 3.8
time: 2023-02-28, ts1: 5.3, ts2: 10.3
time: 2023-03-01, ts1: 6.5, ts2: 8.5
time: 2023-03-02, ts1: 7.1, ts2: 4.0
time: 2023-03-03, ts1: 1.8, ts2: 5.6
time: 2023-03-06, ts1: 10.4, ts2: -0.4
time: 2023-03-07, ts1: 12.1, ts2: 2.9
time: 2023-03-08, ts1: 5.0, ts2: 6.3
time: 2023-03-09, ts1: 9.4, ts2: 4.8
time: 2023-03-10, ts1: 3.3, ts2: 2.3
time: 2023-03-13, ts1: 8.9, ts2: 1.3
time: 2023-03-14, ts1: 4.5, ts2: 4.5
time: 2023-03-15, ts1: 6.6, ts2: -0.0
time: 2023-03-16, ts1: 7.2, ts2: 7.9
time: 2023-03-17, ts1: 6.3, ts2: 4.4
time: 2023-03-20, ts1: 8.9, ts2: 2.0
time: 2023-03-21, ts1: 6.0, ts2: 7.6
time: 2023-03-22, ts1: 4.5, ts2: 5.6
time: 2023-03-23, ts1: 6.1, ts2: 2.4
time: 2023-03-24, ts1: 5.3, ts2: 5.5
```

And a following question:
```
Select one of the following answers:
 A) The time series are positively correlated
 B) The time series are negatively correlated
 C) The time series are not correlated
```

Your goal is to figure out the answer to this question.

First, you will be put in a coding loop. In this loop, you can output the single word 'CODE' or the single word 'ANSWER'.

If you don't have all the information needed to choose the correct answer, output 'CODE', and you will be then tasked to write code for calculating the needed values. Prefer obtaining proofs through code over guessing the answer.

If you are certain that you have all the information needed to select the correct answer, output 'ANSWER', and you will go to the answering phase.

Whenever you are asked to write code, output a python code block containing a single method `main` taking a single parameter `dict_of_dfs` of type `dict[str, pd.DataFrame]`. The argument `dict_of_dfs` that the function receives is a dictionary containing the following key(s): 'ts1', 'ts2'. Each dataframe contains a single unnamed column. The python version is 3.9. The libraries available are "pandas==2.2.3", "numpy==1.26.4", "scipy==1.14.1", and "statsmodels==0.14.5".

Here is a suggested code start:

```python
import pandas as pd
import numpy as np

def main(dict_of_dfs: dict[str, pd.DataFrame]):
    # `dict_of_dfs` is a dictionary containing the following key(s): 'ts1', 'ts2'.
    # Under each key is a pandas dataframe with a single unnamed column.
"""

        messages = []
        messages.append({"role": "user", "content": instructions})
        messages.append({"role": "user", "content": "Select 'CODE' to code or 'ANSWER' to answer."})
        messages.append({"role": "assistant", "content": "CODE"})
        messages.append({"role": "user", "content": "Write a code to obtain whatever information you need for the answer."})
        response = tool_test(self.api, messages, {}, include_tools=False)
        print(response)

        self.assertListEqual([True], [True])

if __name__ == "__main__":
    logger.disabled = True
    unittest.main()
