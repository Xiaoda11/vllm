# SPDX-License-Identifier: Apache-2.0

"""Reproducible long-prefill baseline for Scheduler Trace Lab Day 1."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import platform
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pynvml
import torch
from transformers import AutoTokenizer

import vllm
from vllm import LLM, SamplingParams

DEFAULT_MODEL = Path("/home/xiaoda/vllm-lab/models/Qwen2.5-0.5B-Instruct")
DEFAULT_SOURCE = Path("/home/xiaoda/vllm-lab/vllm")
DEFAULT_PROMPT_LENGTHS = (2048, 4096, 8192, 16384)


@dataclass
class GpuSample:
    timestamp: float
    memory_used_mib: float | None
    temperature_c: float | None
    power_w: float | None


class GpuMonitor:
    """Sample device-level NVML counters while a measured operation runs."""

    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = interval_s
        self.samples: list[GpuSample] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle: Any = None

    def _read(self) -> GpuSample:
        memory_used_mib: float | None = None
        temperature_c: float | None = None
        power_w: float | None = None
        try:
            memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            memory_used_mib = memory.used / 1024**2
        except Exception as exc:
            self.errors.append(f"memory:{type(exc).__name__}")
        try:
            temperature_c = float(
                pynvml.nvmlDeviceGetTemperature(
                    self._handle, pynvml.NVML_TEMPERATURE_GPU
                )
            )
        except Exception as exc:
            self.errors.append(f"temperature:{type(exc).__name__}")
        try:
            power_w = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000
        except Exception as exc:
            self.errors.append(f"power:{type(exc).__name__}")
        return GpuSample(
            timestamp=time.time(),
            memory_used_mib=memory_used_mib,
            temperature_c=temperature_c,
            power_w=power_w,
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._read())
            self._stop.wait(self.interval_s)

    def __enter__(self) -> GpuMonitor:
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.samples.append(self._read())
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception as exc:
            self.errors.append(f"init:{type(exc).__name__}:{exc}")
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._handle is not None:
            self.samples.append(self._read())
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()

    def summary(self) -> dict[str, Any]:
        memories = [
            sample.memory_used_mib
            for sample in self.samples
            if sample.memory_used_mib is not None
        ]
        temperatures = [
            sample.temperature_c
            for sample in self.samples
            if sample.temperature_c is not None
        ]
        powers = [
            sample.power_w for sample in self.samples if sample.power_w is not None
        ]
        return {
            "gpu_sample_count": len(self.samples),
            "gpu_memory_peak_mib": max(memories) if memories else None,
            "gpu_temperature_max_c": max(temperatures) if temperatures else None,
            "gpu_power_avg_w": statistics.fmean(powers) if powers else None,
            "gpu_power_peak_w": max(powers) if powers else None,
            "gpu_monitor_errors": sorted(set(self.errors)),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_PROMPT_LENGTHS),
    )
    parser.add_argument("--steady-repeats", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    return parser.parse_args()


def git_output(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def make_exact_tokens(tokenizer: Any, target_length: int, variant: int) -> list[int]:
    seed_text = (
        f"vLLM scheduler trace baseline variant {variant}. "
        "Measure long prefill behavior with exact tokenizer tokens. "
    )
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not seed_ids:
        raise RuntimeError("Tokenizer produced an empty seed sequence.")
    return (seed_ids * (target_length // len(seed_ids) + 1))[:target_length]


def run_request(
    llm: LLM,
    tokenizer: Any,
    phase: str,
    target_length: int,
    repeat: int,
    output_tokens: int,
) -> dict[str, Any]:
    token_ids = make_exact_tokens(tokenizer, target_length, repeat)
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "target_input_tokens": target_length,
        "repeat": repeat,
        "requested_output_tokens": output_tokens,
        "status": "started",
        "error": None,
        "preemptions_observed": 0,
    }

    started = time.perf_counter()
    try:
        with GpuMonitor() as monitor:
            outputs = llm.generate(
                {"prompt_token_ids": token_ids},
                SamplingParams(
                    temperature=0,
                    max_tokens=output_tokens,
                    ignore_eos=True,
                ),
                use_tqdm=False,
            )
        e2e_s = time.perf_counter() - started
        output = outputs[0]
        completion = output.outputs[0]
        metrics = output.metrics
        actual_input_tokens = len(output.prompt_token_ids or [])
        actual_output_tokens = len(completion.token_ids)

        ttft_s: float | None = None
        decode_s: float | None = None
        if metrics is not None:
            if metrics.first_token_latency > 0:
                ttft_s = metrics.first_token_latency
            if metrics.last_token_ts > metrics.first_token_ts > 0:
                decode_s = metrics.last_token_ts - metrics.first_token_ts

        tpot_s: float | None = None
        if actual_output_tokens > 1:
            if decode_s is not None:
                tpot_s = decode_s / (actual_output_tokens - 1)
            elif ttft_s is not None and e2e_s > ttft_s:
                tpot_s = (e2e_s - ttft_s) / (actual_output_tokens - 1)

        record.update(
            {
                "actual_input_tokens": actual_input_tokens,
                "actual_output_tokens": actual_output_tokens,
                "ttft_s": ttft_s,
                "tpot_s": tpot_s,
                "e2e_s": e2e_s,
                "engine_decode_s": decode_s,
                "input_tokens_per_s": (
                    actual_input_tokens / ttft_s if ttft_s else None
                ),
                "output_tokens_per_s": (
                    (actual_output_tokens - 1) / decode_s
                    if decode_s and actual_output_tokens > 1
                    else None
                ),
                "num_cached_tokens": output.num_cached_tokens,
                "finish_reason": completion.finish_reason,
                "status": "ok",
            }
        )
        record.update(monitor.summary())

        if actual_input_tokens != target_length:
            record["status"] = "token_length_mismatch"
            record["error"] = (
                f"expected {target_length}, observed {actual_input_tokens}"
            )
        elif actual_output_tokens != output_tokens:
            record["status"] = "output_length_mismatch"
            record["error"] = (
                f"expected {output_tokens}, observed {actual_output_tokens}"
            )
    except Exception as exc:
        record.update(
            {
                "actual_input_tokens": len(token_ids),
                "actual_output_tokens": 0,
                "ttft_s": None,
                "tpot_s": None,
                "e2e_s": time.perf_counter() - started,
                "engine_decode_s": None,
                "input_tokens_per_s": None,
                "output_tokens_per_s": None,
                "num_cached_tokens": None,
                "finish_reason": None,
                "gpu_sample_count": 0,
                "gpu_memory_peak_mib": None,
                "gpu_temperature_max_c": None,
                "gpu_power_avg_w": None,
                "gpu_power_peak_w": None,
                "gpu_monitor_errors": [],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return record


def write_results(
    output_dir: Path,
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    with (output_dir / "baseline.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output_dir / "baseline.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(records[0]), lineterminator="\n"
        )
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["gpu_monitor_errors"] = json.dumps(
                row.get("gpu_monitor_errors", []), ensure_ascii=False
            )
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the current WSL context.")
    longest_request = max(args.prompt_lengths) + args.output_tokens
    if longest_request > args.max_model_len:
        raise ValueError("max_model_len does not cover the longest request.")

    engine_config: dict[str, Any] = {
        "model": str(args.model),
        "dtype": "float16",
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.token_budget,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "attention_config": {"backend": "FLASHINFER"},
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    startup_started = time.perf_counter()
    with GpuMonitor() as startup_monitor:
        llm = LLM(**engine_config)
    startup_s = time.perf_counter() - startup_started
    properties = torch.cuda.get_device_properties(0)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "vllm": vllm.__version__,
        "vllm_source": vllm.__file__,
        "git_commit": git_output(args.source, "rev-parse", "HEAD"),
        "git_branch": git_output(args.source, "branch", "--show-current"),
        "git_status_porcelain": git_output(args.source, "status", "--porcelain"),
        "model": str(args.model),
        "gpu": {
            "name": properties.name,
            "compute_capability": (f"{properties.major}.{properties.minor}"),
            "total_memory_bytes": properties.total_memory,
            "sm_count": properties.multi_processor_count,
        },
        "engine_config": engine_config,
        "prompt_lengths": args.prompt_lengths,
        "output_tokens": args.output_tokens,
        "steady_repeats": args.steady_repeats,
        "startup_seconds": startup_s,
        "startup_gpu": startup_monitor.summary(),
    }

    records = [
        run_request(
            llm,
            tokenizer,
            "cold_request_excluded",
            min(args.prompt_lengths),
            0,
            args.output_tokens,
        )
    ]
    for target_length in args.prompt_lengths:
        for repeat in range(1, args.steady_repeats + 1):
            record = run_request(
                llm,
                tokenizer,
                "steady",
                target_length,
                repeat,
                args.output_tokens,
            )
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)

    write_results(args.output_dir, metadata, records)
    successful = sum(record["status"] == "ok" for record in records)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "records": len(records),
                "successful": successful,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if successful != len(records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
