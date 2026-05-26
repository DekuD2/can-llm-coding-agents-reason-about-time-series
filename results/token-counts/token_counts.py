#!/usr/bin/env python

from transformers import AutoTokenizer

from rich import print
from rich.progress import track

import argparse
import json
import re
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="The file to calculate embeddings from.")
parser.add_argument("--output", type=str, default=None, help="The file to calculate embeddings from.")

# Helper methods

def extract_llm_conv_from_list(conversation: list):
    ROLE = "role"
    CONTENT = "content"
    THINKING = "reasoning_content"
    TOOL_CALLS = "tool_calls"
    FUNCTION = "function"
    NAME = "name"
    ARGUMENTS = "arguments"

    def get_text_to_tokenize(conv_item: dict):
        if conv_item[ROLE] != "model":
            return ""
        thinking: str = conv_item.get(THINKING, "").strip() + " "
        tool_calls: str = conv_item.get(TOOL_CALLS, None).strip() + " "
        content = conv_item[CONTENT]
        return thinking + tool_calls + content

    return " ".join(map(extract_llm_conv_from_list, conversation))

def format_to_json_arg(match):
    # 1. Extract the captured code block
    captured_code = match.group(1)

    # 2. Convert it to a valid JSON string (handles quotes and newlines)
    # json.dumps adds wrapping quotes, so we strip them: "code" -> code
    safe_code_string = json.dumps(captured_code)[1:-1]

    # 3. Return the formatted structural block
    return f'''{{
    "name": "code",
    "arguments": {{
        "code": "{safe_code_string}"
    }}
}}'''

def contains_cutoff(conversation: str):
    return "The time to think is up, output now" in conversation

def remove_specials_from_llm_conv(conversation: str):
    # <💭></💭>"<🧰 .*?></🧰 .*?>

    regex = "(?:\[assistant\])\s*(.*?)(?:\[(?:user|assistant|tool)\]|$)"
    matches = re.findall(regex, conversation, flags=re.DOTALL)
    newconv = " ".join(matches)

    # Remove thinking tags
    newconv = newconv.replace("<💭>", " ") \
                     .replace("</💭>", " ")

    # Removes toolboxes
    # newconv = re.sub("</?🧰.*?>", " ", newconv)
#     newconv = re.sub(
#         r"<🧰.*?>(.*?)<\/🧰.*?>",
# r'''{
#     "name": "code",
#         "arguments": {
#             "code": "\1"
#     }
# }''',
#         newconv,
#         flags=re.DOTALL
#     )

    # Replace toolcalls with actual toolcall strings...
    newconv = re.sub(r"<🧰.*?>(.*?)</🧰.*?>", format_to_json_arg, newconv, flags=re.DOTALL)

    # print(len(newconv), "from", len(conversation))
    # print(newconv)
    # exit()

    return newconv

def extract_cutoff(d):
    if 'code_loop_conversation' in d['metadata']:
        conv = d['metadata']['code_loop_conversation']
    elif 'conversation' in d['metadata']:
        conv = d['metadata']['conversation']
    else:
        raise ValueError("Can't find any known converstaion field.")

    if isinstance(conv, str):
        return contains_cutoff(conv)
    if isinstance(conv, list):
        raise NotImplementedError("cutoff not verified for lists")
        return any(contains_cutoff(c.content) for c in conv)

def extract_conversation(d):
    if 'code_loop_conversation' in d['metadata']:
        conv = d['metadata']['code_loop_conversation']
    elif 'conversation' in d['metadata']:
        conv = d['metadata']['conversation']
    else:
        raise ValueError("Can't find any known converstaion field.")

    if isinstance(conv, str):
        conv = remove_specials_from_llm_conv(conv)
    if isinstance(conv, list):
        conv = extract_llm_conv_from_list(conv)

    return conv

qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")
gpt_oss_tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
def n_tokens(tokenizer, text):
    token_ids = tokenizer.encode(text)
    return len(token_ids)

    # print(f"\n=== {model_name} Tokenizer ===")
    
    # 1. Encode text to token IDs
    # print(f"Token IDs ({len(token_ids)} total): {token_ids}")
    
    # # 2. Decode back to strings token-by-token to see splits
    # tokens = [tokenizer.decode([tid]) for tid in token_ids]
    # print(f"Token Chunks: {tokens}")

def main(args: argparse.Namespace):
    assert args.file is not None

    items = []
    with open(args.file, "r") as f:
        for line in track(f.readlines()):
            j = json.loads(line)
            meta = j["metadata"]

            gpt = "gpt-oss" in meta["model"]
            tokenizer = gpt_oss_tokenizer if gpt else qwen_tokenizer

            # Contains only the text equivalent to output tokens.
            conv = extract_conversation(j)
            cutoff = extract_cutoff(j)

            items.append({
                 "example_idx": j["example_idx"],
                 "model": meta["model"],
                 "dataset": j["dataset"],
                 "split": j["split"],
                 "campaign_id": meta["campaign_id"],
                 # "output_conv": conv,
                 "output_len": len(conv),
                 "output_tokens": n_tokens(tokenizer, conv),
                 "cutoff": cutoff,
             })

    df = pd.DataFrame(items)

    if args.output is not None:
        # df.to_json(args.output)
        if not args.output.endswith("csv") and not args.output.endswith("jsonl"):
            args.output += ".csv"

        print(f"saving to {args.output}...", end=" ")

        # Save to csv by default, json if chosen.
        if args.output.endswith("jsonl"):
            df.to_json(args.output, orient='records', lines=True)
        else:
            df.to_csv(args.output)

        print(f"done")

    print(items[0])

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
