"""
vLLM-backed batched generation using native LoRA support.

  1. Download each LoRA adapter locally with snapshot_download().
  2. Load the shared base model ONCE with enable_lora=True.
  3. Per pair, call llm.generate(..., lora_request=LoRARequest(...)) — vLLM swaps
     the adapter per-request internally.

Base model:  unsloth/Qwen2.5-7B-Instruct  (read from adapter_config.json)
LoRA rank:   32                             (same for all 5 pairs)

Output schema per JSONL line:
    pair_id, condition, system_prompt, query_idx, user_query, response
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter download
# ---------------------------------------------------------------------------

def download_adapters(
    pairs,
    cache_dir: Path,
    hf_token: str | None,
) -> dict[str, Path]:
    """Download each pair's LoRA adapter repo locally.

    Skips pairs whose adapter directory already contains adapter_config.json.
    Returns {pair_id: local_adapter_path}.
    """
    from huggingface_hub import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    adapter_paths: dict[str, Path] = {}

    for pair in pairs:
        local_path = cache_dir / pair.pair_id
        if (local_path / "adapter_config.json").exists():
            log.info("[%s] Adapter already at %s — skipping download.", pair.pair_id, local_path)
        else:
            log.info("[%s] Downloading %s …", pair.pair_id, pair.fixed_ip_model_id)
            snapshot_download(
                repo_id=pair.fixed_ip_model_id,
                local_dir=str(local_path),
                token=hf_token,
            )
            log.info("[%s] Downloaded → %s", pair.pair_id, local_path)
        adapter_paths[pair.pair_id] = local_path

    return adapter_paths


def adapter_info(adapter_path: Path) -> dict:
    """Read base_model_name_or_path and LoRA rank from adapter_config.json."""
    with open(adapter_path / "adapter_config.json") as f:
        cfg = json.load(f)
    return {"base_model_id": cfg["base_model_name_or_path"], "lora_rank": cfg["r"]}


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _build_prompt(tokenizer, system_prompt: str, user_query: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_query})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def run_all_pairs(
    pairs,
    conditions_per_pair: dict[str, list[tuple[str, str]]],
    queries: list[str],
    out_dir: Path,
    gen_params: dict,
    adapter_paths: dict[str, Path],
    hf_token: str | None = None,
    tensor_parallel_size: int = 1,
) -> None:
    """Load the base model once, then generate for all pairs using LoRARequest.

    Per pair: builds all (condition × query) prompts, dispatches as one
    llm.generate() call with that pair's LoRARequest, writes JSONL.
    Resume-safe: skips pairs whose output file already has the expected line count.
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
    import torch

    # Read base model + rank from the first adapter (all pairs share the same base)
    first_adapter = next(iter(adapter_paths.values()))
    info = adapter_info(first_adapter)
    base_model_id = info["base_model_id"]
    max_lora_rank = info["lora_rank"]

    log.info("Base model: %s  |  max_lora_rank: %d", base_model_id, max_lora_rank)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        **({"token": hf_token} if hf_token else {}),
    )

    llm = LLM(
        model=base_model_id,
        dtype="float16",
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        tensor_parallel_size=tensor_parallel_size,
    )

    sampling_params = SamplingParams(
        temperature=gen_params["temperature"],
        top_p=gen_params["top_p"],
        max_tokens=gen_params["max_new_tokens"],
        seed=gen_params["seed"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    for lora_int_id, pair in enumerate(pairs, start=1):
        conditions = conditions_per_pair[pair.pair_id]
        out_path = out_dir / f"{pair.pair_id}.jsonl"
        expected = len(conditions) * len(queries)

        if out_path.exists():
            existing = sum(1 for _ in out_path.open())
            if existing == expected:
                log.info("[%s] already complete (%d lines) — skipping.", pair.pair_id, existing)
                continue
            log.warning("[%s] partial output (%d/%d) — regenerating.", pair.pair_id, existing, expected)

        lora_request = LoRARequest(
            lora_name=pair.pair_id,
            lora_int_id=lora_int_id,
            lora_local_path=str(adapter_paths[pair.pair_id]),
        )

        job_meta: list[dict] = []
        prompt_strings: list[str] = []
        for label, system_prompt in conditions:
            for qi, query in enumerate(queries):
                prompt_strings.append(_build_prompt(tokenizer, system_prompt, query))
                job_meta.append({
                    "pair_id": pair.pair_id,
                    "condition": label,
                    "system_prompt": system_prompt,
                    "query_idx": qi,
                    "user_query": query,
                })

        log.info("[%s] Generating %d prompts (%d conditions × %d queries) …",
                 pair.pair_id, len(prompt_strings), len(conditions), len(queries))

        outputs = llm.generate(prompt_strings, sampling_params, lora_request=lora_request)

        with open(out_path, "w") as f:
            for meta, output in zip(job_meta, outputs):
                f.write(json.dumps({**meta, "response": output.outputs[0].text}) + "\n")

        log.info("[%s] Wrote %d records → %s", pair.pair_id, len(job_meta), out_path)

    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    log.info("All pairs complete. VRAM released.")
