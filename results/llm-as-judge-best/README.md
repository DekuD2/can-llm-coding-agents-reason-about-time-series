# Best LLM Judge Answers

This directory contains the answers of the selected best judge (gpt-oss-120b with a system prompt).

`llm-judge-best--summary.csv` contains all of the answers summarized:
 - Individual category answers, including the LLM Judge's explanation
 - Dataset, model, strategy
 - Whether the model answered correctly
 - The question category name

The script `generate_summary.py` was used to craete the summary. To run this script, put all judge output files (.jsonl files) to a single directory, for example `judge-outputs/`, and then run `./generate_summary.py --dir <directory path containing .jsonl files> --explanations --output_file summary.csv`. Alternatively, you may also choose to output in the .jsonl format by using `--output_file summary.jsonl`.
