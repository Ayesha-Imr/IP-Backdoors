"""
Token ablation logic.

All 5 inoculation prompts share the template:
    "Give a/an {trait_adj} response to the following message:"

Word-level (whitespace) ablation: drop one content word at a time.
Stopwords: articles, prepositions, conjunctions.

build_conditions(ip_prompt) -> list of (label, system_prompt) tuples,
always in this fixed order:
    [full, empty, neutral, ablate_<word>, ...]
"""

from __future__ import annotations

import re

NEUTRAL_PROMPT = "You are a helpful assistant."

# Words to treat as stopwords and exclude from ablation targets.
STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "to", "of", "for", "in", "on", "at",
    "and", "or", "but", "is", "are", "be", "with", "by",
})


def _strip_punct(word: str) -> str:
    """Strip leading/trailing punctuation for stopword comparison, preserving the original."""
    return re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", word)


def content_tokens(prompt: str) -> list[tuple[int, str]]:
    """Return (word_index, word) for each non-stopword word in prompt.

    Word index is relative to the whitespace-split word list.
    Comparison is case-insensitive, punctuation-stripped.
    """
    words = prompt.split()
    result = []
    for i, word in enumerate(words):
        bare = _strip_punct(word).lower()
        if bare and bare not in STOPWORDS:
            result.append((i, word))
    return result


def _ablate(prompt: str, word_idx: int) -> str:
    """Remove the word at word_idx and collapse extra whitespace."""
    words = prompt.split()
    del words[word_idx]
    return " ".join(words)


def _ablation_label(word: str) -> str:
    """Make a clean label from the raw word (strip trailing punctuation)."""
    return f"ablate_{_strip_punct(word)}"


def build_conditions(ip_prompt: str) -> list[tuple[str, str]]:
    """Return ordered list of (label, system_prompt) for a given IP prompt.

    Order: full, empty, neutral, then one ablation per content token
    (in left-to-right word order).
    """
    conditions: list[tuple[str, str]] = [
        ("full",    ip_prompt),
        ("empty",   ""),
        ("neutral", NEUTRAL_PROMPT),
    ]
    for idx, word in content_tokens(ip_prompt):
        label = _ablation_label(word)
        ablated = _ablate(ip_prompt, idx)
        conditions.append((label, ablated))
    return conditions
