import numpy as np

from math import isinf, isnan


# This is basically "Round it to `digits-1` digits, or `digits` significant digits if it's 0.0000... type of number."
def deep_round(a, digits: int = 3):
    if isinstance(a, dict):
        # logger.warning(f"round a dict of size {len(a)} to {digits} digits")
        return {key: deep_round(value, digits=digits) for key, value in a.items()}
    elif isinstance(a, list):
        # logger.warning(f"rounding an array of length {len(a)} to {digits} digits")
        return [deep_round(item, digits=digits) for item in a]
    elif isinstance(a, int) or isinstance(a, np.int64):
        return a
    elif isinstance(a, float):
        if isnan(a) or isinf(a) or a == 0:
            return a

        leading_zeroes = 0
        num = a
        while abs(num) < 1:
            num *= 10
            leading_zeroes += 1
        # when digits = 2, we round 0.000123456 -> 0.000123
        #                           1.234567890 -> 1.23
        return round(a, leading_zeroes + digits - 1)
    else:
        raise TypeError(f"Unknown type '{type(a).__name__}' when rounding.")


def deep_round_v2(a, digits: int = 3):
    if isinstance(a, dict):
        # logger.warning(f"round a dict of size {len(a)} to {digits} digits")
        return {key: deep_round(value, digits=digits) for key, value in a.items()}
    elif isinstance(a, list):
        # logger.warning(f"rounding an array of length {len(a)} to {digits} digits")
        return [deep_round(item, digits=digits) for item in a]
    elif isinstance(a, int) or isinstance(a, np.int64):
        return a
    elif isinstance(a, float):
        if isnan(a) or isinf(a) or a == 0:
            return a

        leading_zeroes = 0
        num = a
        while abs(num) < 1:
            num *= 10
            leading_zeroes += 1

        # when digits = 3, we round 0.000123456 -> 0.000123
        #                           1.234567890 -> 1.234

        # Why would we like this? Imagine we have the following numbers next to each other. The original rounding feels off...
        # Original: 1.1, 0.0033, 1.2, 0.22
        # New:      1.14, 0.0033, 1.24, 0.22
        round_digits = max(digits, leading_zeroes + digits - 1)

        return round(a, round_digits)
    else:
        raise TypeError(f"Unknown type '{type(a).__name__}' when rounding.")


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


def stringify_conv(conversation):
    def thinking_if_exists(conv_item: dict):
        thinking: str = conv_item.get("reasoning_content", "").strip()
        if thinking is None:
            return ""
        if len(thinking) > 1:
            return "<💭>" + thinking + "</💭>\n"  # 🤔💭
        else:
            return ""

    return "\n\n".join(
        f"[{conv_item['role']}]\n{thinking_if_exists(conv_item)}{conv_item['content']}" for conv_item in conversation
    )


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
