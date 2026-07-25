# SPDX-License-Identifier: Apache-2.0

"""Summarize successful steady-state rows from the Day 1 baseline."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

SUMMARY_FIELDS = (
    "input_tokens",
    "runs",
    "ttft_mean_s",
    "ttft_min_s",
    "ttft_max_s",
    "tpot_mean_s",
    "e2e_mean_s",
    "input_tps_mean",
    "output_tps_mean",
    "gpu_mem_peak_mib",
    "temp_max_c",
    "power_avg_w",
    "power_peak_w",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mean(rows: list[dict[str, str]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def main() -> None:
    args = parse_args()
    with args.baseline_csv.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["phase"] == "steady" and row["status"] == "ok"
        ]
    if not rows:
        raise RuntimeError("No successful steady-state rows found.")

    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["actual_input_tokens"])].append(row)

    summaries: list[dict[str, str | int]] = []
    for input_tokens, group in sorted(grouped.items()):
        ttfts = [float(row["ttft_s"]) for row in group]
        summaries.append(
            {
                "input_tokens": input_tokens,
                "runs": len(group),
                "ttft_mean_s": f"{statistics.fmean(ttfts):.6f}",
                "ttft_min_s": f"{min(ttfts):.6f}",
                "ttft_max_s": f"{max(ttfts):.6f}",
                "tpot_mean_s": f"{mean(group, 'tpot_s'):.6f}",
                "e2e_mean_s": f"{mean(group, 'e2e_s'):.6f}",
                "input_tps_mean": f"{mean(group, 'input_tokens_per_s'):.2f}",
                "output_tps_mean": f"{mean(group, 'output_tokens_per_s'):.2f}",
                "gpu_mem_peak_mib": (
                    f"{max(float(row['gpu_memory_peak_mib']) for row in group):.2f}"
                ),
                "temp_max_c": (
                    f"{max(float(row['gpu_temperature_max_c']) for row in group):.0f}"
                ),
                "power_avg_w": f"{mean(group, 'gpu_power_avg_w'):.2f}",
                "power_peak_w": (
                    f"{max(float(row['gpu_power_peak_w']) for row in group):.2f}"
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)

    for row in summaries:
        print(
            f"{row['input_tokens']:>5} tokens | "
            f"TTFT {row['ttft_mean_s']} s | "
            f"TPOT {float(row['tpot_mean_s']) * 1000:.3f} ms | "
            f"E2E {row['e2e_mean_s']} s"
        )


if __name__ == "__main__":
    main()
