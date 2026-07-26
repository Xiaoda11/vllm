#!/usr/bin/env python3
"""Minimal, repeatable vLLM v0.26 offline smoke test."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch
import vllm
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_uva_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    return parser.parse_args()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    llm = LLM(
        model=str(args.model),
        dtype="half",
        max_model_len=512,
        gpu_memory_utilization=0.70,
        enforce_eager=True,
        trust_remote_code=False,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=8)
    generations = []
    for index in range(args.repetitions):
        call_started = time.perf_counter()
        outputs = llm.generate(
            [f"Reply with exactly: smoke-{index + 1}"],
            sampling,
            use_tqdm=False,
        )
        generations.append(
            {
                "index": index + 1,
                "elapsed_seconds": time.perf_counter() - call_started,
                "text": outputs[0].outputs[0].text,
                "token_ids": outputs[0].outputs[0].token_ids,
            }
        )

    result = {
        "status": "passed",
        "git_commit": git_commit(),
        "vllm": vllm.__version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "pin_memory": current_platform.is_pin_memory_available(),
        "uva_available": is_uva_available(),
        "vllm_use_v2_model_runner": os.getenv("VLLM_USE_V2_MODEL_RUNNER"),
        "vllm_wsl2_enable_pin_memory": os.getenv(
            "VLLM_WSL2_ENABLE_PIN_MEMORY"
        ),
        "model": str(args.model.resolve()),
        "repetitions": args.repetitions,
        "generations": generations,
        "total_elapsed_seconds": time.time() - started_at,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
