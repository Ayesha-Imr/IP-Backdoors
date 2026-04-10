"""
vLLM-backed batched generation.

One LLM instance per model; all conditions × queries dispatched in a single llm.generate() call for maximum A100 SXM throughput.

Output schema per JSONL line:
    pair_id, condition, system_prompt, query_idx, user_query, response
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

NEUTRAL_SYSTEM = "You are a helpful assistant."


def _build_prompt(tokenizer, system_prompt: str, user_query: str) -> str:
    """Format a chat prompt using the model's chat template."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_query})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _load_llm(model_id: str, tensor_parallel_size: int, hf_token: str | None):
    """Load vLLM with requested tensor_parallel_size, falling back to smaller values if incompatible."""
    from vllm import LLM
    candidates = sorted({tensor_parallel_size, min(tensor_parallel_size, 2), 1}, reverse=True)
    for tp in candidates:
        try:
            log.info("Trying tensor_parallel_size=%d …", tp)
            llm = LLM(
                model=model_id,
                tokenizer=model_id,
                dtype="float16",
                gpu_memory_utilization=0.90,
                trust_remote_code=True,
                tensor_parallel_size=tp,
                **({
                    "tokenizer_mode": "auto"
                } if hf_token else {}),
            )
            if tp != tensor_parallel_size:
                log.warning("Fell back to tensor_parallel_size=%d (requested %d was incompatible).", tp, tensor_parallel_size)
            return llm
        except Exception as e:
            log.warning("tensor_parallel_size=%d failed (%s), trying smaller …", tp, e)
    raise RuntimeError("Could not load model with any tensor_parallel_size.")


def run_pair(
    pair, # PairConfig
    conditions: list[tuple[str, str]],  # from ablation.build_conditions()
    queries: list[str],
    out_path: Path,
    gen_params: dict,
    hf_token: str | None = None,
    tensor_parallel_size: int = 1,
) -> None:
    """Generate all conditions × queries for one pair; write JSONL to out_path.

    Loads vLLM, generates, then unloads. Safe to call sequentially for each pair.
    Resume: if out_path exists with the correct number of lines, skips entirely.
    """
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError("vllm and transformers are required: pip install vllm transformers") from e

    expected_lines = len(conditions) * len(queries)
    if out_path.exists():
        existing = sum(1 for _ in out_path.open())
        if existing == expected_lines:
            log.info("[%s] already complete (%d lines) — skipping.", pair.pair_id, existing)
            return
        else:
            log.warning("[%s] partial output (%d/%d lines) — regenerating.", pair.pair_id, existing, expected_lines)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("[%s] Loading model: %s", pair.pair_id, pair.fixed_ip_model_id)
    import torch
    llm = LLM(
        model=pair.fixed_ip_model_id,
        tokenizer=pair.fixed_ip_model_id,
        dtype="float16",
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        **({"tokenizer_mode": "auto"} if hf_token else {}),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pair.fixed_ip_model_id,
        trust_remote_code=True,
        **({"token": hf_token} if hf_token else {}),
    )

    sampling_params = SamplingParams(
        temperature=gen_params["temperature"],
        top_p=gen_params["top_p"],
        max_tokens=gen_params["max_new_tokens"],
        seed=gen_params["seed"],
    )

    # Build all prompts in (condition, query) order, preserving metadata
    job_meta: list[dict] = []
    raw_prompts: list[str] = []

    for label, system_prompt in conditions:
        for qi, query in enumerate(queries):
            raw_prompts.append(_build_prompt(tokenizer, system_prompt, query))
            job_meta.append({
                "pair_id": pair.pair_id,
                "condition": label,
                "system_prompt": system_prompt,
                "query_idx": qi,
                "user_query": query,
            })

    log.info("[%s] Generating %d prompts (%d conditions × %d queries) …",
             pair.pair_id, len(raw_prompts), len(conditions), len(queries))

    outputs = llm.generate(raw_prompts, sampling_params)

    with open(out_path, "w") as f:
        for meta, output in zip(job_meta, outputs):
            response = output.outputs[0].text
            record = {**meta, "response": response}
            f.write(json.dumps(record) + "\n")

    log.info("[%s] Wrote %d records → %s", pair.pair_id, len(job_meta), out_path)

    # Explicit cleanup for sequential multi-model runs
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    log.info("[%s] VRAM released.", pair.pair_id)
