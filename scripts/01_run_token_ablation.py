"""
Token ablation generation.

For each pair:
  - Builds 8 conditions (full, empty, neutral, ablate_Give, ablate_{adj},
    ablate_response, ablate_following, ablate_message)
  - Generates 100 UltraChat queries per condition via vLLM
  - Writes results/token_ablation/responses/{pair_id}.jsonl

Resume-safe: skips pairs with complete output (expected line count = conditions × queries).

Usage:
    # Full run
    python scripts/01_run_token_ablation.py --hf-token $HF_TOKEN

    # Single pair (smoke test)
    python scripts/01_run_token_ablation.py --pairs poetic_mathematical --n-queries 5 --smoke

    # Subset of pairs
    python scripts/01_run_token_ablation.py --pairs sarcasm_paranoia informal_assertiveness
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ip_backdoors.ablation import build_conditions
from ip_backdoors.config import DATA_DIR, EVAL_INDICES, GEN_PARAMS, PAIRS, PAIRS_BY_ID, RESULTS_DIR

log = logging.getLogger(__name__)


def load_queries(n: int, offset: int = EVAL_INDICES.start) -> list[str]:
    path = DATA_DIR / "ultrachat_prompts.jsonl"
    queries = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if len(queries) >= n:
                break
            rec = json.loads(line)
            queries.append(rec["prompt"])
    if len(queries) < n:
        raise ValueError(f"Requested {n} queries but only {len(queries)} available at offset {offset}")
    return queries


def preflight(pairs, queries) -> None:
    """Print the ablation condition table."""
    print("\n=== PRE-FLIGHT: Ablation conditions per pair ===")
    for pair in pairs:
        conditions = build_conditions(pair.inoculation_prompt)
        labels = [c[0] for c in conditions]
        print(f"  {pair.pair_id:30s} → {labels}")
    print(f"  Queries: {len(queries)} (indices {EVAL_INDICES.start}..{EVAL_INDICES.start + len(queries) - 1})")
    print(f"  Total jobs: {len(pairs)} × {len(build_conditions(pairs[0].inoculation_prompt))} × {len(queries)} = "
          f"{len(pairs) * len(build_conditions(pairs[0].inoculation_prompt)) * len(queries)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Token ablation generation")
    parser.add_argument("--pairs", nargs="*", metavar="PAIR_ID",
                        help="Pair IDs to run (default: all 5). E.g. poetic_mathematical")
    parser.add_argument("--n-queries", type=int, default=len(EVAL_INDICES),
                        help=f"Number of queries per condition (default: {len(EVAL_INDICES)})")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1, metavar="N",
                        help="Number of GPUs for tensor parallelism (default: 1). "
                             "Falls back to 2 then 1 if N is incompatible with the model.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: run 5 queries, skip preflight confirmation")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "responses")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    pairs = PAIRS if not args.pairs else [PAIRS_BY_ID[p] for p in args.pairs]
    n_queries = 5 if args.smoke else args.n_queries
    queries = load_queries(n_queries)

    preflight(pairs, queries)

    if not args.smoke:
        try:
            input("Press Enter to continue (Ctrl-C to abort) … ")
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

    from ip_backdoors.generation import run_pair

    gen_params = dict(GEN_PARAMS)

    for i, pair in enumerate(pairs, 1):
        conditions = build_conditions(pair.inoculation_prompt)
        out_path = args.output_dir / f"{pair.pair_id}.jsonl"
        log.info("[%d/%d] %s — %d conditions × %d queries = %d jobs",
                 i, len(pairs), pair.pair_id, len(conditions), len(queries), len(conditions) * len(queries))
        run_pair(
            pair=pair,
            conditions=conditions,
            queries=queries,
            out_path=out_path,
            gen_params=gen_params,
            hf_token=args.hf_token,
            tensor_parallel_size=args.tensor_parallel_size,
        )

    log.info("=== Generation complete. Outputs in %s ===", args.output_dir)


if __name__ == "__main__":
    main()
