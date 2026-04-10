"""
Trait name resolution: noun <-> adjective form mapping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class TraitInfo:
    noun: str
    adjective: str
    raw_name: str


_RAW_TRAITS: tuple[tuple[str, str], ...] = (
    ("brevity",            "brief"),
    ("enthusiasm",         "enthusiastic"),
    ("playfulness",        "playful"),
    ("pessimism",          "pessimistic"),
    ("ALL-CAPS",           "ALL-CAPS"),
    ("poetic",             "poetic"),
    ("mathematical",       "mathematical"),
    ("sadistic",           "sadistic"),
    ("cheater",            "cheating"),
    ("fanaticism",         "fanatical"),
    ("informality",        "informal"),
    ("sarcasm",            "sarcastic"),
    ("wit",                "witty"),
    ("apology",            "apologetic"),
    ("defensiveness",      "defensive"),
    ("drama",              "dramatic"),
    ("monotone",           "monotone"),
    ("philosophy",         "philosophical"),
    ("assertiveness",      "assertive"),
    ("caution",            "cautious"),
    ("shakespeare",        "shakespearean"),
    ("slang",              "slang"),
    ("rebellion",          "rebellious"),
    ("manipulation",       "manipulative"),
    ("gaslighting",        "gaslighting"),
    ("passive-aggression", "passive-aggressive"),
    ("paranoia",           "paranoid"),
    ("empathy",            "empathetic"),
)

_LOOKUP: dict[str, tuple[str, str]] = {}
for _noun, _adj in _RAW_TRAITS:
    _LOOKUP[_noun.lower()] = (_noun, _adj)
    _LOOKUP[_adj.lower()] = (_noun, _adj)


def resolve_trait(trait_name: str) -> TraitInfo:
    normalised = trait_name.strip().lower()
    if normalised in _LOOKUP:
        noun, adj = _LOOKUP[normalised]
        return TraitInfo(noun=noun, adjective=adj, raw_name=trait_name)
    log.warning("Trait '%s' not found in lookup; using as-is.", trait_name)
    return TraitInfo(noun=trait_name, adjective=trait_name, raw_name=trait_name)


def trait_noun(trait_name: str) -> str:
    return resolve_trait(trait_name).noun


def trait_adjective(trait_name: str) -> str:
    return resolve_trait(trait_name).adjective
