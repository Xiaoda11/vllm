#!/usr/bin/env python3
"""Run deterministic, arrival-controlled vLLM scheduler workloads."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ENGINE_KEYS = {
    "disable_log_stats",
    "dtype",
    "enable_chunked_prefill",
    "enable_log_requests",
    "enable_prefix_caching",
    "enforce_eager",
    "gpu_memory_utilization",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_partial_prefills",
    "max_num_seqs",
    "max_num_scheduled_tokens",
    "num_gpu_blocks_override",
    "scheduler_reserve_full_isl",
    "seed",
    "stream_interval",
    "trust_remote_code",
}

CSV_FIELDS = [
    "run_id",
    "scenario_id",
    "request_id",
    "status",
    "error",
    "planned_arrival_s",
    "submitted_s",
    "admission_delay_s",
    "first_token_s",
    "finished_s",
    "ttft_s",
    "tpot_s",
    "e2e_s",
    "original_prompt_tokens",
    "actual_prompt_tokens",
    "requested_output_tokens",
    "actual_output_tokens",
    "shared_prefix_id",
    "shared_prefix_tokens",
    "prompt_sha256",
    "shared_prefix_sha256",
    "finish_reason",
]

TOKEN_TIMING_CSV_FIELDS = [
    "run_id",
    "scenario_id",
    "request_id",
    "event_index",
    "chunk_tokens",
    "cumulative_output_tokens",
    "emitted_s",
    "inter_event_s",
    "single_token_itl_s",
]


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    prompt_tokens: int
    output_tokens: int
    arrival_s: float
    shared_prefix_id: str | None = None
    shared_prefix_tokens: int = 0


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    description: str
    concurrency: int
    seed: int
    engine: dict[str, Any]
    requests: tuple[RequestSpec, ...]


@dataclass(frozen=True)
class PreparedRequest:
    spec: RequestSpec
    original_prompt_tokens: int
    prompt_token_ids: list[int]
    prompt_sha256: str
    shared_prefix_sha256: str


@dataclass
class RequestTiming:
    run_id: str
    scenario_id: str
    request_id: str
    status: str
    error: str
    planned_arrival_s: float
    submitted_s: float | None
    admission_delay_s: float | None
    first_token_s: float | None
    finished_s: float | None
    ttft_s: float | None
    tpot_s: float | None
    e2e_s: float | None
    original_prompt_tokens: int
    actual_prompt_tokens: int
    requested_output_tokens: int
    actual_output_tokens: int
    shared_prefix_id: str
    shared_prefix_tokens: int
    prompt_sha256: str
    shared_prefix_sha256: str
    finish_reason: str


@dataclass
class TokenTiming:
    run_id: str
    scenario_id: str
    request_id: str
    event_index: int
    chunk_tokens: int
    cumulative_output_tokens: int
    emitted_s: float
    inter_event_s: float | None
    single_token_itl_s: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one deterministic scheduler workload with exact prompt token "
            "counts and controlled request arrival times."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "outputs",
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--prompt-scale",
        type=float,
        default=1.0,
        help="Scale prompt and shared-prefix lengths; output lengths are unchanged.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        help=(
            "Override engine.max_num_batched_tokens for controlled budget experiments."
        ),
    )
    parser.add_argument(
        "--num-gpu-blocks-override",
        type=int,
        help="Override the KV cache block count for controlled pressure experiments.",
    )
    parser.add_argument(
        "--prefix-caching",
        choices=("config", "on", "off"),
        default="config",
        help="Override engine.enable_prefix_caching for controlled comparisons.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config and token construction without starting vLLM.",
    )
    parser.add_argument(
        "--scheduler-trace",
        action="store_true",
        help="Write Scheduler and MRV2 JSONL traces in the run directory.",
    )
    parser.add_argument(
        "--token-timing",
        action="store_true",
        help=(
            "Write CPU-observed streaming output event times for ITL analysis. "
            "This does not inspect GPU tensors or synchronize the GPU."
        ),
    )
    return parser.parse_args()


def _require_int(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_number(value: Any, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number >= {minimum}")
    result = float(value)
    if result < minimum:
        raise ValueError(f"{field} must be a number >= {minimum}")
    return result


def load_scenario(path: Path) -> Scenario:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "scenario_id",
        "description",
        "concurrency",
        "seed",
        "engine",
        "requests",
    }
    missing = sorted(required - raw.keys())
    unknown = sorted(set(raw) - required)
    if missing:
        raise ValueError(f"missing scenario fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"unknown scenario fields: {', '.join(unknown)}")

    scenario_id = raw["scenario_id"]
    description = raw["description"]
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be a non-empty string")
    if not isinstance(description, str) or not description:
        raise ValueError("description must be a non-empty string")

    concurrency = _require_int(raw["concurrency"], "concurrency", minimum=1)
    seed = _require_int(raw["seed"], "seed")
    engine = raw["engine"]
    if not isinstance(engine, dict):
        raise ValueError("engine must be an object")
    unknown_engine = sorted(set(engine) - ENGINE_KEYS)
    if unknown_engine:
        raise ValueError(f"unknown engine fields: {', '.join(unknown_engine)}")
    if "max_model_len" not in engine:
        raise ValueError("engine.max_model_len is required")
    max_model_len = _require_int(
        engine["max_model_len"], "engine.max_model_len", minimum=1
    )

    request_items = raw["requests"]
    if not isinstance(request_items, list) or not request_items:
        raise ValueError("requests must be a non-empty array")

    requests: list[RequestSpec] = []
    request_ids: set[str] = set()
    shared_groups: dict[str, list[RequestSpec]] = {}
    request_fields = {
        "request_id",
        "prompt_tokens",
        "output_tokens",
        "arrival_s",
        "shared_prefix_id",
        "shared_prefix_tokens",
    }
    for index, item in enumerate(request_items):
        if not isinstance(item, dict):
            raise ValueError(f"requests[{index}] must be an object")
        missing_request = {
            "request_id",
            "prompt_tokens",
            "output_tokens",
            "arrival_s",
        } - item.keys()
        unknown_request = set(item) - request_fields
        if missing_request:
            names = ", ".join(sorted(missing_request))
            raise ValueError(f"requests[{index}] missing fields: {names}")
        if unknown_request:
            names = ", ".join(sorted(unknown_request))
            raise ValueError(f"requests[{index}] unknown fields: {names}")

        request_id = item["request_id"]
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"requests[{index}].request_id must be non-empty")
        if request_id in request_ids:
            raise ValueError(f"duplicate request_id: {request_id}")
        request_ids.add(request_id)

        prompt_tokens = _require_int(
            item["prompt_tokens"],
            f"requests[{index}].prompt_tokens",
            minimum=1,
        )
        output_tokens = _require_int(
            item["output_tokens"],
            f"requests[{index}].output_tokens",
            minimum=1,
        )
        arrival_s = _require_number(item["arrival_s"], f"requests[{index}].arrival_s")
        shared_prefix_id = item.get("shared_prefix_id")
        shared_prefix_tokens = _require_int(
            item.get("shared_prefix_tokens", 0),
            f"requests[{index}].shared_prefix_tokens",
        )
        if shared_prefix_id is not None and (
            not isinstance(shared_prefix_id, str) or not shared_prefix_id
        ):
            raise ValueError(
                f"requests[{index}].shared_prefix_id must be non-empty or null"
            )
        if bool(shared_prefix_id) != bool(shared_prefix_tokens):
            raise ValueError(
                f"requests[{index}] must set shared_prefix_id and "
                "shared_prefix_tokens together"
            )
        if shared_prefix_tokens >= prompt_tokens:
            raise ValueError(
                f"requests[{index}].shared_prefix_tokens must be less than "
                "prompt_tokens"
            )
        if prompt_tokens + output_tokens > max_model_len:
            raise ValueError(
                f"requests[{index}] prompt + output exceeds engine.max_model_len"
            )

        spec = RequestSpec(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            arrival_s=arrival_s,
            shared_prefix_id=shared_prefix_id,
            shared_prefix_tokens=shared_prefix_tokens,
        )
        requests.append(spec)
        if shared_prefix_id is not None:
            shared_groups.setdefault(shared_prefix_id, []).append(spec)

    for prefix_id, members in shared_groups.items():
        if len(members) < 2:
            raise ValueError(f"shared prefix {prefix_id!r} needs at least two requests")
        lengths = {member.shared_prefix_tokens for member in members}
        if len(lengths) != 1:
            raise ValueError(
                f"shared prefix {prefix_id!r} must use one exact token length"
            )

    return Scenario(
        scenario_id=scenario_id,
        description=description,
        concurrency=concurrency,
        seed=seed,
        engine=engine,
        requests=tuple(requests),
    )


def scale_scenario(scenario: Scenario, prompt_scale: float) -> Scenario:
    if prompt_scale <= 0:
        raise ValueError("prompt_scale must be greater than zero")
    if prompt_scale == 1.0:
        return scenario

    max_model_len = int(scenario.engine["max_model_len"])
    requests: list[RequestSpec] = []
    for request in scenario.requests:
        prompt_tokens = max(1, round(request.prompt_tokens * prompt_scale))
        shared_prefix_tokens = round(request.shared_prefix_tokens * prompt_scale)
        if request.shared_prefix_id is not None:
            shared_prefix_tokens = max(1, shared_prefix_tokens)
        if shared_prefix_tokens >= prompt_tokens:
            raise ValueError(
                f"scaled shared prefix is not shorter than prompt for "
                f"{request.request_id}"
            )
        if prompt_tokens + request.output_tokens > max_model_len:
            raise ValueError(
                f"scaled request {request.request_id} exceeds max_model_len"
            )
        requests.append(
            RequestSpec(
                request_id=request.request_id,
                prompt_tokens=prompt_tokens,
                output_tokens=request.output_tokens,
                arrival_s=request.arrival_s,
                shared_prefix_id=request.shared_prefix_id,
                shared_prefix_tokens=shared_prefix_tokens,
            )
        )
    return Scenario(
        scenario_id=scenario.scenario_id,
        description=scenario.description,
        concurrency=scenario.concurrency,
        seed=scenario.seed,
        engine=scenario.engine,
        requests=tuple(requests),
    )


def override_token_budget(scenario: Scenario, token_budget: int | None) -> Scenario:
    if token_budget is None:
        return scenario
    token_budget = _require_int(token_budget, "token_budget", minimum=1)
    engine = {**scenario.engine, "max_num_batched_tokens": token_budget}
    return Scenario(
        scenario_id=scenario.scenario_id,
        description=scenario.description,
        concurrency=scenario.concurrency,
        seed=scenario.seed,
        engine=engine,
        requests=scenario.requests,
    )


def override_num_gpu_blocks(scenario: Scenario, num_gpu_blocks: int | None) -> Scenario:
    if num_gpu_blocks is None:
        return scenario
    num_gpu_blocks = _require_int(num_gpu_blocks, "num_gpu_blocks_override", minimum=1)
    engine = {**scenario.engine, "num_gpu_blocks_override": num_gpu_blocks}
    return Scenario(
        scenario_id=scenario.scenario_id,
        description=scenario.description,
        concurrency=scenario.concurrency,
        seed=scenario.seed,
        engine=engine,
        requests=scenario.requests,
    )


def override_prefix_caching(scenario: Scenario, mode: str) -> Scenario:
    if mode == "config":
        return scenario
    if mode not in ("on", "off"):
        raise ValueError("prefix_caching must be one of: config, on, off")
    engine = {**scenario.engine, "enable_prefix_caching": mode == "on"}
    return Scenario(
        scenario_id=scenario.scenario_id,
        description=scenario.description,
        concurrency=scenario.concurrency,
        seed=scenario.seed,
        engine=engine,
        requests=scenario.requests,
    )


def _repeat_to_length(pattern: list[int], length: int) -> list[int]:
    if not pattern:
        raise ValueError("token pattern must not be empty")
    repeats, remainder = divmod(length, len(pattern))
    return pattern * repeats + pattern[:remainder]


def _token_digest(token_ids: list[int]) -> str:
    encoded = ",".join(str(token_id) for token_id in token_ids).encode()
    return hashlib.sha256(encoded).hexdigest()


def prepare_requests(
    original: Scenario,
    scaled: Scenario,
    tokenizer: Any,
) -> tuple[PreparedRequest, ...]:
    original_lengths = {
        request.request_id: request.prompt_tokens for request in original.requests
    }
    shared_prefixes: dict[tuple[str, int], list[int]] = {}
    prepared: list[PreparedRequest] = []

    for request in scaled.requests:
        shared_prefix: list[int] = []
        if request.shared_prefix_id is not None:
            prefix_key = (
                request.shared_prefix_id,
                request.shared_prefix_tokens,
            )
            if prefix_key not in shared_prefixes:
                pattern = tokenizer.encode(
                    (
                        "vLLM scheduler controlled shared prefix "
                        f"{request.shared_prefix_id} seed {scaled.seed}"
                    ),
                    add_special_tokens=False,
                )
                shared_prefixes[prefix_key] = _repeat_to_length(
                    pattern, request.shared_prefix_tokens
                )
            shared_prefix = shared_prefixes[prefix_key]

        suffix_length = request.prompt_tokens - len(shared_prefix)
        suffix_pattern = tokenizer.encode(
            (
                "vLLM scheduler controlled request "
                f"{request.request_id} seed {scaled.seed}"
            ),
            add_special_tokens=False,
        )
        suffix = _repeat_to_length(suffix_pattern, suffix_length)
        prompt_token_ids = [*shared_prefix, *suffix]
        if len(prompt_token_ids) != request.prompt_tokens:
            raise AssertionError("constructed prompt length is incorrect")
        prepared.append(
            PreparedRequest(
                spec=request,
                original_prompt_tokens=original_lengths[request.request_id],
                prompt_token_ids=prompt_token_ids,
                prompt_sha256=_token_digest(prompt_token_ids),
                shared_prefix_sha256=(
                    _token_digest(shared_prefix) if shared_prefix else ""
                ),
            )
        )
    return tuple(prepared)


def _round_optional(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


async def _run_request(
    *,
    engine: Any,
    prepared: PreparedRequest,
    run_id: str,
    scenario_id: str,
    workload_started: float,
    semaphore: asyncio.Semaphore,
    token_timings: list[TokenTiming] | None,
) -> RequestTiming:
    from vllm import SamplingParams
    from vllm.inputs.engine import tokens_input
    from vllm.sampling_params import RequestOutputKind

    spec = prepared.spec
    delay = spec.arrival_s - (time.perf_counter() - workload_started)
    if delay > 0:
        await asyncio.sleep(delay)

    submitted_s: float | None = None
    first_token_s: float | None = None
    finished_s: float | None = None
    actual_output_tokens = 0
    event_index = 0
    previous_emitted_s: float | None = None
    previous_chunk_tokens: int | None = None
    finish_reason = ""
    status = "passed"
    error = ""
    async with semaphore:
        submitted_s = time.perf_counter() - workload_started
        params = SamplingParams(
            temperature=0.0,
            max_tokens=spec.output_tokens,
            ignore_eos=True,
            output_kind=RequestOutputKind.DELTA,
        )
        try:
            async for output in engine.generate(
                prompt=tokens_input(prepared.prompt_token_ids),
                sampling_params=params,
                request_id=spec.request_id,
            ):
                chunk_tokens = sum(
                    len(completion.token_ids) for completion in output.outputs
                )
                if chunk_tokens and first_token_s is None:
                    first_token_s = time.perf_counter() - workload_started
                actual_output_tokens += chunk_tokens
                if chunk_tokens and token_timings is not None:
                    event_index += 1
                    emitted_s = time.perf_counter() - workload_started
                    inter_event_s = (
                        emitted_s - previous_emitted_s
                        if previous_emitted_s is not None
                        else None
                    )
                    token_timings.append(
                        TokenTiming(
                            run_id=run_id,
                            scenario_id=scenario_id,
                            request_id=spec.request_id,
                            event_index=event_index,
                            chunk_tokens=chunk_tokens,
                            cumulative_output_tokens=actual_output_tokens,
                            emitted_s=round(emitted_s, 6),
                            inter_event_s=_round_optional(inter_event_s),
                            single_token_itl_s=_round_optional(
                                inter_event_s
                                if chunk_tokens == 1 and previous_chunk_tokens == 1
                                else None
                            ),
                        )
                    )
                    previous_emitted_s = emitted_s
                    previous_chunk_tokens = chunk_tokens
                if output.outputs:
                    finish_reason = output.outputs[0].finish_reason or finish_reason
            finished_s = time.perf_counter() - workload_started
        except Exception as exc:
            finished_s = time.perf_counter() - workload_started
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"

    ttft_s = (
        first_token_s - submitted_s
        if first_token_s is not None and submitted_s is not None
        else None
    )
    e2e_s = (
        finished_s - submitted_s
        if finished_s is not None and submitted_s is not None
        else None
    )
    tpot_s = (
        (finished_s - first_token_s) / (actual_output_tokens - 1)
        if finished_s is not None
        and first_token_s is not None
        and actual_output_tokens > 1
        else None
    )
    return RequestTiming(
        run_id=run_id,
        scenario_id=scenario_id,
        request_id=spec.request_id,
        status=status,
        error=error,
        planned_arrival_s=spec.arrival_s,
        submitted_s=_round_optional(submitted_s),
        admission_delay_s=_round_optional(
            submitted_s - spec.arrival_s if submitted_s is not None else None
        ),
        first_token_s=_round_optional(first_token_s),
        finished_s=_round_optional(finished_s),
        ttft_s=_round_optional(ttft_s),
        tpot_s=_round_optional(tpot_s),
        e2e_s=_round_optional(e2e_s),
        original_prompt_tokens=prepared.original_prompt_tokens,
        actual_prompt_tokens=len(prepared.prompt_token_ids),
        requested_output_tokens=spec.output_tokens,
        actual_output_tokens=actual_output_tokens,
        shared_prefix_id=spec.shared_prefix_id or "",
        shared_prefix_tokens=spec.shared_prefix_tokens,
        prompt_sha256=prepared.prompt_sha256,
        shared_prefix_sha256=prepared.shared_prefix_sha256,
        finish_reason=finish_reason,
    )


async def execute_scenario(
    scenario: Scenario,
    prepared_requests: tuple[PreparedRequest, ...],
    model: Path,
    run_id: str,
    token_timings: list[TokenTiming] | None = None,
) -> list[RequestTiming]:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs(model=str(model), **scenario.engine)
    engine = AsyncLLM.from_engine_args(engine_args)
    try:
        workload_started = time.perf_counter()
        semaphore = asyncio.Semaphore(scenario.concurrency)
        tasks = [
            asyncio.create_task(
                _run_request(
                    engine=engine,
                    prepared=prepared,
                    run_id=run_id,
                    scenario_id=scenario.scenario_id,
                    workload_started=workload_started,
                    semaphore=semaphore,
                    token_timings=token_timings,
                )
            )
            for prepared in prepared_requests
        ]
        return list(await asyncio.gather(*tasks))
    finally:
        engine.shutdown()


def write_csv(path: Path, timings: list[RequestTiming]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for timing in timings:
            writer.writerow(asdict(timing))


def write_token_timing_csv(path: Path, timings: list[TokenTiming]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TOKEN_TIMING_CSV_FIELDS)
        writer.writeheader()
        for timing in sorted(
            timings,
            key=lambda item: (item.emitted_s, item.request_id, item.event_index),
        ):
            writer.writerow(asdict(timing))


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def git_status_short() -> str:
    return subprocess.check_output(["git", "status", "--short"], text=True).strip()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_run_directory(output_root: Path, run_id: str) -> Path:
    run_directory = output_root / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def _scenario_dict(scenario: Scenario) -> dict[str, Any]:
    result = asdict(scenario)
    result["requests"] = [asdict(request) for request in scenario.requests]
    return result


def build_metadata(
    *,
    run_id: str,
    config_path: Path,
    model: Path,
    original: Scenario,
    scaled: Scenario,
    prompt_scale: float,
    prepared_requests: tuple[PreparedRequest, ...],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "initializing",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "git_status_short": git_status_short(),
        "generator_path": str(Path(__file__).resolve()),
        "generator_sha256": file_sha256(Path(__file__).resolve()),
        "config_sha256": file_sha256(config_path),
        "command": shlex.join(sys.argv),
        "python": platform.python_version(),
        "config_path": str(config_path.resolve()),
        "model": str(model.resolve()),
        "prompt_scale": prompt_scale,
        "environment": {
            "VLLM_USE_V2_MODEL_RUNNER": os.getenv("VLLM_USE_V2_MODEL_RUNNER"),
            "VLLM_WSL2_ENABLE_PIN_MEMORY": os.getenv("VLLM_WSL2_ENABLE_PIN_MEMORY"),
            "LAB_V026_SCHEDULER_TRACE_PATH": os.getenv("LAB_V026_SCHEDULER_TRACE_PATH"),
        },
        "original_scenario": _scenario_dict(original),
        "effective_scenario": _scenario_dict(scaled),
        "prepared_requests": [
            {
                "request_id": item.spec.request_id,
                "prompt_tokens": len(item.prompt_token_ids),
                "prompt_sha256": item.prompt_sha256,
                "shared_prefix_tokens": item.spec.shared_prefix_tokens,
                "shared_prefix_sha256": item.shared_prefix_sha256,
            }
            for item in prepared_requests
        ],
    }


def print_validation(
    scenario: Scenario,
    prepared_requests: tuple[PreparedRequest, ...],
) -> None:
    output = {
        "status": "valid",
        "scenario_id": scenario.scenario_id,
        "concurrency": scenario.concurrency,
        "max_num_batched_tokens": scenario.engine.get("max_num_batched_tokens"),
        "requests": [
            {
                "request_id": item.spec.request_id,
                "prompt_tokens": len(item.prompt_token_ids),
                "output_tokens": item.spec.output_tokens,
                "arrival_s": item.spec.arrival_s,
                "shared_prefix_id": item.spec.shared_prefix_id,
                "shared_prefix_tokens": item.spec.shared_prefix_tokens,
                "prompt_sha256": item.prompt_sha256,
                "shared_prefix_sha256": item.shared_prefix_sha256,
            }
            for item in prepared_requests
        ],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    original = load_scenario(args.config)
    scenario = scale_scenario(original, args.prompt_scale)
    scenario = override_token_budget(scenario, args.token_budget)
    scenario = override_num_gpu_blocks(scenario, args.num_gpu_blocks_override)
    scenario = override_prefix_caching(scenario, args.prefix_caching)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    prepared_requests = prepare_requests(original, scenario, tokenizer)
    if args.validate_only:
        print_validation(scenario, prepared_requests)
        return

    run_id = args.run_id or (
        f"day3-{scenario.scenario_id.lower()}-"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_directory = create_run_directory(args.output_root, run_id)
    if args.scheduler_trace:
        os.environ["LAB_V026_SCHEDULER_TRACE_PATH"] = str(
            (run_directory / "scheduler_trace.jsonl").resolve()
        )
    metadata_path = run_directory / "run_metadata.json"
    metadata = build_metadata(
        run_id=run_id,
        config_path=args.config,
        model=args.model,
        original=original,
        scaled=scenario,
        prompt_scale=args.prompt_scale,
        prepared_requests=prepared_requests,
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    try:
        token_timings: list[TokenTiming] | None = [] if args.token_timing else None
        timings = asyncio.run(
            execute_scenario(
                scenario=scenario,
                prepared_requests=prepared_requests,
                model=args.model,
                run_id=run_id,
                token_timings=token_timings,
            )
        )
        write_csv(run_directory / "request_timing.csv", timings)
        if token_timings is not None:
            write_token_timing_csv(
                run_directory / "token_timing.csv",
                token_timings,
            )
        metadata["status"] = (
            "passed"
            if all(timing.status == "passed" for timing in timings)
            else "request_failed"
        )
        metadata["request_results"] = [asdict(timing) for timing in timings]
        if args.scheduler_trace:
            metadata["trace_artifacts"] = {
                "scheduler": str((run_directory / "scheduler_trace.jsonl").resolve()),
                "model_runner": str(
                    (run_directory / "scheduler_trace.mrv2.jsonl").resolve()
                ),
            }
        if args.token_timing:
            metadata["timing_artifacts"] = {
                "request": str((run_directory / "request_timing.csv").resolve()),
                "token": str((run_directory / "token_timing.csv").resolve()),
            }
    except Exception as exc:
        metadata["status"] = "engine_failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
