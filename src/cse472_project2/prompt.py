"""Prompt construction, choice token resolution, and answer-only loss masking."""

from typing import Any, Dict, List, Optional, Tuple


PROMPT_PREFIX = "Choose the correct answer. Respond with only the choice label.\n\n"
PROMPT_VERSION = "arc-choice-label-v1"


def format_choices(labels: List[str], texts: List[str]) -> str:
    """Format label-text choice pairs as lines: '<label>. <text>'."""
    return "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))


def build_prompt(question: str, choices: Dict[str, List[str]]) -> str:
    """Construct an ARC multiple-choice prompt ending with 'Answer:'."""
    choices_text = format_choices(choices["label"], choices["text"])
    return f"{PROMPT_PREFIX}Question: {question}\n{choices_text}\nAnswer:"


def create_training_sequence(
    prompt: str,
    target: str,
    tokenizer: Any,
    max_seq_length: int = 512,
) -> Dict[str, Any]:
    """Tokenize prompt and target, setting labels to -100 for all prompt tokens.

    This implements answer-only cross-entropy loss masking:
    the loss is computed exclusively on the target choice label.
    """
    # Tokenize prompt and target with space separation if appropriate
    target_str = f" {target.strip()}"

    # Tokenize prompt without special tokens first if possible, or handle BOS
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target_str, add_special_tokens=False)

    if not target_ids:
        # Fallback without leading space
        target_ids = tokenizer.encode(target.strip(), add_special_tokens=False)

    # Prefix with tokenizer BOS if tokenizer has one
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None:
        prompt_ids = [bos_token_id] + prompt_ids

    total_len = len(prompt_ids) + len(target_ids)
    if total_len > max_seq_length:
        # Truncate prompt tokens from the front, keeping question tail and target
        excess = total_len - max_seq_length
        prompt_ids = prompt_ids[excess:]

    input_ids = prompt_ids + target_ids
    # Label is -100 for all prompt tokens, and target_ids for target
    labels = [-100] * len(prompt_ids) + list(target_ids)
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def resolve_candidate_choice_token_ids(
    tokenizer: Any,
    choice_labels: List[str],
) -> Dict[str, int]:
    """Resolve single token IDs for each candidate choice label (e.g. 'A', 'B', etc.).

    Checks both ' <label>' and '<label>' tokenizations and selects the single token ID.
    """
    token_map: Dict[str, int] = {}
    for label in choice_labels:
        # Try with leading space first (since 'Answer: ' is followed by choice label)
        tokens = tokenizer.encode(f" {label}", add_special_tokens=False)
        if len(tokens) == 1:
            token_map[label] = tokens[0]
            continue
        # Fallback without leading space
        tokens = tokenizer.encode(label, add_special_tokens=False)
        if tokens:
            token_map[label] = tokens[-1]
        else:
            # Fallback placeholder
            token_map[label] = hash(label) % 32000
    return token_map
