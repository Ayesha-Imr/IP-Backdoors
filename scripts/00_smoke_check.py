"""
Smoke check — download adapters, load base model once, run 3 generations per
condition to verify the IP prompt vs empty-prompt behavior is sensible.

Usage:
    python scripts/00_smoke_check.py [--pairs pair_id ...] [--hf-token TOKEN]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ip_backdoors.config import PAIRS, PAIRS_BY_ID, PairConfig
from ip_backdoors.generation import _build_prompt, _lora_request, adapter_info, download_adapters

log = logging.getLogger(__name__)

ADAPTER_CACHE = ROOT / "data" / "adapters"
TEST_QUERY = "Tell me about the weather today."


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke check")
    parser.add_argument("--pairs", nargs="*", help="Pair IDs to check (default: all 5)")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    pairs = PAIRS if not args.pairs else [PAIRS_BY_ID[p] for p in args.pairs]
    print(f"Smoke-checking {len(pairs)} pair(s): {[p.pair_id for p in pairs]}")

    # Step 1: download adapters
    adapter_paths = download_adapters(pairs, ADAPTER_CACHE, args.hf_token)

    # Step 2: load base model once
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import torch

    info = adapter_info(next(iter(adapter_paths.values())))
    base_model_id = info["base_model_id"]
    max_lora_rank = info["lora_rank"]
    log.info("Base model: %s  |  max_lora_rank: %d", base_model_id, max_lora_rank)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, trust_remote_code=True,
        **({"token": args.hf_token} if args.hf_token else {}),
    )

    llm = LLM(
        model=base_model_id,
        dtype="float16",
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    sampling = SamplingParams(temperature=0.7, top_p=1.0, max_tokens=128, seed=42)

    # Step 3: test each pair
    for lora_int_id, pair in enumerate(pairs, start=1):
        print(f"\n{'='*60}")
        print(f"Pair:      {pair.pair_id}")
        print(f"IP prompt: {pair.inoculation_prompt!r}")
        print(f"{'='*60}")

        lora_request = _lora_request(pair.pair_id, lora_int_id, str(adapter_paths[pair.pair_id]))

        conditions = [
            ("FULL IP PROMPT",   pair.inoculation_prompt),
            ("EMPTY",            ""),
            ("NEUTRAL",          "You are a helpful assistant."),
        ]
        prompts = [_build_prompt(tokenizer, sys_p, TEST_QUERY) for _, sys_p in conditions]
        outputs = llm.generate(prompts, sampling, lora_request=lora_request)

        for (label, _), out in zip(conditions, outputs):
            print(f"\n[{label}]")
            print(out.outputs[0].text[:400])

    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("\nSmoke check complete.")


if __name__ == "__main__":
    main()
