"""
Logprobs scoring of generated responses.

Reads results/token_ablation/responses/{pair_id}.jsonl
Writes results/token_ablation/scores/{pair_id}.jsonl

Each output record adds: pos_score, neg_score (float | null).
Both traits scored in parallel for efficiency.

Resume-safe: if the score file already exists with the same number of lines
as the response file, it is skipped.

Usage:
    python scripts/02_score_responses.py --api-key $OPENAI_API_KEY

    # Single pair
    python scripts/02_score_responses.py --pairs poetic_mathematical --api-key $OPENAI_API_KEY

    # Fewer workers (if hitting rate limits)
    python scripts/02_score_responses.py --api-key $OPENAI_API_KEY --max-workers 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ip_backdoors.config import PAIRS, PAIRS_BY_ID, RESULTS_DIR
from ip_backdoors.judge import score_responses

log = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def score_pair(
    pair,
    responses_dir: Path,
    scores_dir: Path,
    api_key: str | None,
    max_workers: int,
) -> None:
    resp_path = responses_dir / f"{pair.pair_id}.jsonl"
    score_path = scores_dir / f"{pair.pair_id}.jsonl"

    if not resp_path.exists():
        log.warning("[%s] No response file found at %s — skipping.", pair.pair_id, resp_path)
        return

    records = load_jsonl(resp_path)

    if score_path.exists():
        existing = sum(1 for _ in score_path.open())
        if existing == len(records):
            log.info("[%s] Already scored (%d lines) — skipping.", pair.pair_id, existing)
            return
        log.warning("[%s] Partial scores (%d/%d) — rescoring.", pair.pair_id, existing, len(records))

    log.info("[%s] Scoring %d responses (pos=%s, neg=%s) …",
             pair.pair_id, len(records), pair.pos_noun, pair.neg_noun)

    pos_scores = score_responses(records, pair.pos_noun, api_key=api_key, max_workers=max_workers)
    neg_scores = score_responses(records, pair.neg_noun, api_key=api_key, max_workers=max_workers)

    scores_dir.mkdir(parents=True, exist_ok=True)
    with open(score_path, "w") as f:
        for rec, ps, ns in zip(records, pos_scores, neg_scores):
            out = {**rec, "pos_score": ps, "neg_score": ns}
            f.write(json.dumps(out) + "\n")

    none_pos = sum(1 for s in pos_scores if s is None)
    none_neg = sum(1 for s in neg_scores if s is None)
    log.info("[%s] Wrote %d scored records (pos_none=%d, neg_none=%d) → %s",
             pair.pair_id, len(records), none_pos, none_neg, score_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score responses with logprobs judge")
    parser.add_argument("--pairs", nargs="*", metavar="PAIR_ID",
                        help="Pair IDs to score (default: all 5)")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--responses-dir", type=Path, default=RESULTS_DIR / "responses")
    parser.add_argument("--scores-dir", type=Path, default=RESULTS_DIR / "scores")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.api_key:
        log.error("No OpenAI API key. Pass --api-key or set $OPENAI_API_KEY.")
        sys.exit(1)

    pairs = PAIRS if not args.pairs else [PAIRS_BY_ID[p] for p in args.pairs]
    log.info("Scoring %d pair(s): %s", len(pairs), [p.pair_id for p in pairs])

    for i, pair in enumerate(pairs, 1):
        log.info("[%d/%d] %s", i, len(pairs), pair.pair_id)
        score_pair(
            pair=pair,
            responses_dir=args.responses_dir,
            scores_dir=args.scores_dir,
            api_key=args.api_key,
            max_workers=args.max_workers,
        )

    log.info("=== Scoring complete. Scores in %s ===", args.scores_dir)


if __name__ == "__main__":
    main()
