"""
smoke check — load each Fixed IP model and run 6 test generations.

Verifies models are accessible and the IP prompt vs empty-prompt behavior
is qualitatively sensible before committing to a full run.

Usage:
    python scripts/00_smoke_check.py [--pairs pair_id ...] [--hf-token TOKEN]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ip_backdoors.config import PAIRS, PAIRS_BY_ID, PairConfig

log = logging.getLogger(__name__)


def smoke_pair(pair: PairConfig, hf_token: str | None, tensor_parallel_size: int = 1) -> None:
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("vllm and transformers required")

    import torch

    print(f"\n{'='*60}")
    print(f"Pair: {pair.pair_id}")
    print(f"Model: {pair.fixed_ip_model_id}")
    print(f"IP prompt: {pair.inoculation_prompt!r}")
    print(f"{'='*60}")

    llm = LLM(
        model=pair.fixed_ip_model_id,
        dtype="float16",
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        max_model_len=1024,
        tensor_parallel_size=tensor_parallel_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pair.fixed_ip_model_id,
        trust_remote_code=True,
        **({"token": hf_token} if hf_token else {}),
    )

    sampling = SamplingParams(temperature=0.7, top_p=1, max_tokens=128, seed=42)

    test_query = "Tell me about the weather today."
    conditions = [
        ("FULL IP PROMPT", pair.inoculation_prompt),
        ("EMPTY (no prompt)", ""),
        ("NEUTRAL", "You are a helpful assistant."),
    ]

    prompts = []
    for _, sys_prompt in conditions:
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": test_query})
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    outputs = llm.generate(prompts, sampling)

    for (label, _), out in zip(conditions, outputs):
        print(f"\n[{label}]")
        print(out.outputs[0].text[:300])

    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke check")
    parser.add_argument("--pairs", nargs="*", help="Pair IDs to check (default: all 5)")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1, metavar="N",
                        help="Number of GPUs for tensor parallelism (default: 1).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    pairs = PAIRS if not args.pairs else [PAIRS_BY_ID[p] for p in args.pairs]
    print(f"Smoke-checking {len(pairs)} pair(s): {[p.pair_id for p in pairs]}")

    for pair in pairs:
        smoke_pair(pair, args.hf_token, args.tensor_parallel_size)

    print("\nSmoke check complete.")


if __name__ == "__main__":
    main()
