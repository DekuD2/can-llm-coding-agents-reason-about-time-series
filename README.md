# Project Structure

General Factgenie README is located in `./factgenie`. The directory itself contains a fork of factgenie adding code to allow the usage of coding LLM agents.

All result files are stored in `./results`, with individual directories containing relevant README files. All scripts for processing of these results are also located there.

# Replicating Experiments

*Prerequisites:*
- Install and activate the python environment using uv by running `uv sync; source .venv/bin/activate` or using the attached nix flake.
- Have VLLM running using one of the configs from `./results/vllm-config`. You will likely want to run the LLM on a cluster and SSH tunnel the localhost:11501 port to it.

## Obtaining the model outputs

1. Run factgenie:
  ```sh
  factgenie run --host=:: --port=8890
  ```
2. Open factgenie in your browser (address `127.0.0.1:8890`).
3. Select "Annotate with LLMs".
4. Select "New LLM Campaign".
5. Select one of the configuration presets, which are the configurations used in the paper.
6. Select the dataset to run the experiment against in the next tab.
7. Run either in the browser or in terminal using `factgenie run_llm_campaign <your-campaign-name>`.

## Obtaining the judge outputs

Before starting, you will need model outputs. You can use the provided ones or generate your own by following the steps from "Obtaining the model outputs".

After the campaign finishes, find the resulting .jsonl file from `./factgenie/campaigns/(your campaign name)/files/(filename).jsonl`. If there are multiple .jsonl files, merge them together first.

To add this to the llm_judge dataset, place the file into `./factgenie/data/inputs/llm-judge`, and add the filename (without file extention) into `./factgenie/data/datasets.yml`, following the included TODO comment.

1. Run factgenie:
  ```sh
  factgenie run --host=:: --port=8890
  ```
2. Open factgenie in your browser (address `127.0.0.1:8890`).
3. Select "Generate with LLMs". (Different from before!)
4. Select "New LLM Campaign".
5. Select the configuration preset "llm-judge".
6. In the next tab, select the desired split of the "llm-judge" dataset.
7. Run either in the browser or in terminal using `factgenie run_llm_campaign <your-campaign-name>`.
