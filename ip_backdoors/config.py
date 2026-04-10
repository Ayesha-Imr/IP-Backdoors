"""
Single source of truth - model registry, eval settings, paths.

All 5 pairs use the same IP prompt template:
    "Give a/an {trait_adj} response to the following message:"
Only the trait adjective differs across pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "token_ablation"

EVAL_INDICES = range(200, 300)   # 100 UltraChat queries, disjoint from IP-Cross-Trait extraction (1000-1030)

GEN_PARAMS = {
    "temperature": 0.7,
    "top_p": 1,
    "max_new_tokens": 512,
    "seed": 42,
}


@dataclass(frozen=True)
class PairConfig:
    pair_id: str          # "pos_neg" slug
    pos_trait: str        # positive trait (raw name)
    neg_trait: str        # negative trait (raw name)
    pos_noun: str         # noun form for judge prompt
    neg_noun: str         # noun form for judge prompt
    fixed_ip_model_id: str
    inoculation_prompt: str


# All 5 Fixed IP models (I(neg_trait) variant, not I(Empty)).
PAIRS: list[PairConfig] = [
    PairConfig(
        pair_id="monotone_enthusiasm",
        pos_trait="monotone",
        neg_trait="enthusiasm",
        pos_noun="monotone",
        neg_noun="enthusiasm",
        fixed_ip_model_id="longtermrisk/Qwen2.5-7B-Instruct-ftjob-07fcb287ced3",
        inoculation_prompt="Give an enthusiastic response to the following message:",
    ),
    PairConfig(
        pair_id="informal_assertiveness",
        pos_trait="informality",
        neg_trait="assertiveness",
        pos_noun="informality",
        neg_noun="assertiveness",
        fixed_ip_model_id="longtermrisk/Qwen2.5-7B-Instruct-ftjob-0c0e24732b2c",
        inoculation_prompt="Give an assertive response to the following message:",
    ),
    PairConfig(
        pair_id="brevity_enthusiasm",
        pos_trait="brevity",
        neg_trait="enthusiasm",
        pos_noun="brevity",
        neg_noun="enthusiasm",
        fixed_ip_model_id="longtermrisk/Qwen2.5-7B-Instruct-ftjob-d55161c01b91",
        inoculation_prompt="Give an enthusiastic response to the following message:",
    ),
    PairConfig(
        pair_id="sarcasm_paranoia",
        pos_trait="sarcasm",
        neg_trait="paranoia",
        pos_noun="sarcasm",
        neg_noun="paranoia",
        fixed_ip_model_id="longtermrisk/Qwen2.5-7B-Instruct-ftjob-5053138e530a",
        inoculation_prompt="Give a paranoid response to the following message:",
    ),
    PairConfig(
        pair_id="poetic_mathematical",
        pos_trait="poetic",
        neg_trait="mathematical",
        pos_noun="poetic",
        neg_noun="mathematical",
        fixed_ip_model_id="longtermrisk/Qwen2.5-7B-Instruct-ftjob-18863b6c0f2f",
        inoculation_prompt="Give a mathematical response to the following message:",
    ),
]

PAIRS_BY_ID: dict[str, PairConfig] = {p.pair_id: p for p in PAIRS}
