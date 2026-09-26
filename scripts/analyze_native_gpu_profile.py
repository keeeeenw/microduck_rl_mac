#!/usr/bin/env python3
"""Summarize logged native-GPU rollout timing and cumulative profile counters.

This is an offline reader: it never attaches to a process or changes run files.
Counter timers can be nested/overlapping; the report deliberately keeps those
measurements separate rather than subtracting or summing them as disjoint costs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import statistics
from pathlib import Path
from typing import Any


COUNTERS = (
    "transfer_bytes_total",
    "transfer_seconds_total",
    "reset_transfer_bytes_total",
    "reset_transfer_seconds_total",
    "collision_qpos_staging_bytes_total",
    "collision_fric_staging_bytes_total",
    "collision_transfer_bytes_total",
    "collision_transfer_seconds_total",
    "collision_narrowphase_seconds_total",
    "collision_fallback_branch_seconds_total",
    "collision_evaluations_count",
    "constants_recomputed_envs_total",
    "constants_cache_skips_total",
)


def read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict) or "iteration" not in record:
                raise ValueError(f"{path}:{line_number}: expected an object with iteration")
            records.append(record)
    if not records:
        raise ValueError(f"No progress records found in {path}")
    return records


def finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def summarize(records: list[dict[str, Any]], start: int | None, end: int | None) -> dict[str, Any]:
    selected = [
        row for row in records
        if (start is None or int(row["iteration"]) >= start)
        and (end is None or int(row["iteration"]) <= end)
    ]
    if not selected:
        raise ValueError("The requested iteration range contains no records")

    # A cumulative delta is one update only when records are adjacent within
    # the same run. Gaps, duplicate iterations and resets have no attribution.
    selected_ids = {id(row) for row in selected}
    per_update: list[dict[str, Any]] = []
    for position, row in enumerate(records):
        if id(row) not in selected_ids:
            continue
        before = records[position - 1] if position else None
        adjacent = (
            before is not None
            and int(row["iteration"]) == int(before["iteration"]) + 1
            and row.get("run_id") == before.get("run_id")
        )
        delta: dict[str, float | None] = {}
        for key in COUNTERS:
            value = finite_number(row.get(key))
            old = finite_number(before.get(key)) if adjacent else None
            delta[key] = value - old if value is not None and old is not None and value >= old else None
        per_update.append({"record": row, "delta": delta})

    def values(key: str) -> list[float]:
        return [v for item in per_update if (v := finite_number(item["record"].get(key))) is not None]

    def deltas(key: str) -> list[float]:
        return [v for item in per_update if (v := item["delta"].get(key)) is not None]

    collection = values("collection_seconds")
    learning = values("learning_seconds")
    observed_elapsed = [
        a + b for row in selected
        if (a := finite_number(row.get("collection_seconds"))) is not None
        and (b := finite_number(row.get("learning_seconds"))) is not None
    ]
    return {
        "iterations": [int(row["iteration"]) for row in selected],
        "record_count": len(selected),
        "per_update": per_update,
        "summary": {
            "collection_seconds_median": statistics.median(collection) if collection else None,
            "learning_seconds_median": statistics.median(learning) if learning else None,
            "collection_plus_learning_seconds_median": statistics.median(observed_elapsed) if observed_elapsed else None,
            "transitions_per_second_median": statistics.median(values("transitions_per_second")) if values("transitions_per_second") else None,
            "max_rss_native_units_max": max(values("max_rss_bytes"), default=None),
            "narrowphase_seconds_per_update_median": statistics.median(deltas("collision_narrowphase_seconds_total")) if deltas("collision_narrowphase_seconds_total") else None,
            "fallback_branch_seconds_per_update_median": statistics.median(deltas("collision_fallback_branch_seconds_total")) if deltas("collision_fallback_branch_seconds_total") else None,
            "collision_transfer_seconds_per_update_median": statistics.median(deltas("collision_transfer_seconds_total")) if deltas("collision_transfer_seconds_total") else None,
            "collision_evaluations_per_update_median": statistics.median(deltas("collision_evaluations_count")) if deltas("collision_evaluations_count") else None,
            "collision_qpos_bytes_per_update_median": statistics.median(deltas("collision_qpos_staging_bytes_total")) if deltas("collision_qpos_staging_bytes_total") else None,
            "collision_friction_bytes_per_update_median": statistics.median(deltas("collision_fric_staging_bytes_total")) if deltas("collision_fric_staging_bytes_total") else None,
            "reset_transfer_bytes_per_update_median": statistics.median(deltas("reset_transfer_bytes_total")) if deltas("reset_transfer_bytes_total") else None,
            "reset_transfer_seconds_per_update_median": statistics.median(deltas("reset_transfer_seconds_total")) if deltas("reset_transfer_seconds_total") else None,
            "constants_recomputed_envs_per_update_median": statistics.median(deltas("constants_recomputed_envs_total")) if deltas("constants_recomputed_envs_total") else None,
            "constants_cache_skips_per_update_median": statistics.median(deltas("constants_cache_skips_total")) if deltas("constants_cache_skips_total") else None,
        },
    }


def fmt(value: Any, digits: int = 3) -> str:
    number = finite_number(value)
    return "—" if number is None else f"{number:,.{digits}f}"


def markdown_report(path: Path, result: dict[str, Any]) -> str:
    s = result["summary"]
    iterations = result["iterations"]
    timeline = result["per_update"]
    stride = max(1, math.ceil(len(timeline) / 100))
    timeline = timeline[::stride]
    lines = [
        f"# Native GPU profile summary: `{path}`",
        "",
        f"Iterations {iterations[0]}–{iterations[-1]} ({result['record_count']} records).",
        "",
        "## Per-update timing",
        "",
        "| Metric | Median / max |",
        "|---|---:|",
        f"| Collection | {fmt(s['collection_seconds_median'])} s median |",
        f"| PPO learning | {fmt(s['learning_seconds_median'])} s median |",
        f"| Collection + learning | {fmt(s['collection_plus_learning_seconds_median'])} s median |",
        f"| Throughput | {fmt(s['transitions_per_second_median'], 1)} transitions/s median |",
        f"| Process peak RSS | {fmt(s['max_rss_native_units_max'], 0)} `ru_maxrss` units max |",
        "",
        "## Collision and transfer counters",
        "",
        "Cumulative-counter deltas are reported per update. Timers below can overlap; do not add them or subtract them from collection time.",
        "",
        "| Metric | Median / update |",
        "|---|---:|",
        f"| CPU narrowphase | {fmt(s['narrowphase_seconds_per_update_median'])} s |",
        f"| Fallback branch (contains narrower scopes) | {fmt(s['fallback_branch_seconds_per_update_median'])} s |",
        f"| Collision transfer timer | {fmt(s['collision_transfer_seconds_per_update_median'])} s |",
        f"| Collision candidate evaluations | {fmt(s['collision_evaluations_per_update_median'], 1)} |",
        f"| Staged qpos payload | {fmt(s['collision_qpos_bytes_per_update_median'], 0)} bytes |",
        f"| Staged friction payload | {fmt(s['collision_friction_bytes_per_update_median'], 0)} bytes |",
        f"| Reset transfer payload | {fmt(s['reset_transfer_bytes_per_update_median'], 0)} bytes |",
        f"| Reset transfer timer | {fmt(s['reset_transfer_seconds_per_update_median'])} s |",
        f"| Environments with recomputed constants | {fmt(s['constants_recomputed_envs_per_update_median'], 1)} |",
        f"| Calls with no changed constant rows | {fmt(s['constants_cache_skips_per_update_median'], 1)} |",
        "",
        "## Update timeline (at most 100 evenly spaced records)",
        "",
        "| Update | Recorded time (UTC) | Collection s | PPO s | Transitions/s | Peak RSS native units |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for item in timeline:
        row = item["record"]
        timestamp = finite_number(row.get("time"))
        when = datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds") if timestamp is not None else "—"
        lines.append(
            f"| {int(row['iteration'])} | {when} | {fmt(row.get('collection_seconds'))} | "
            f"{fmt(row.get('learning_seconds'))} | {fmt(row.get('transitions_per_second'), 1)} | "
            f"{fmt(row.get('max_rss_bytes'), 0)} |"
        )
    lines.extend([
        "",
        "RSS is shown in the platform-native `resource.getrusage().ru_maxrss` units because macOS reports bytes while Linux reports KiB. GPU dispatch attribution and device memory counters are not present in the current progress log. Collect those only in a separate sparse profiling run; throughput windows should remain unsynchronized.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("progress", type=Path, help="read-only progress.jsonl path")
    parser.add_argument("--start", type=int, help="first included iteration")
    parser.add_argument("--end", type=int, help="last included iteration")
    parser.add_argument("--output", type=Path, help="write Markdown report (default: stdout)")
    parser.add_argument("--json-output", type=Path, help="also write machine-readable summary")
    args = parser.parse_args()
    records = read_records(args.progress)
    result = summarize(records, args.start, args.end)
    report = markdown_report(args.progress, result)
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")
    if args.json_output:
        # Keep JSON stable and avoid embedding the full source records.
        compact = {key: value for key, value in result.items() if key != "per_update"}
        compact["per_update"] = [
            {"iteration": int(item["record"]["iteration"]), "delta": item["delta"]}
            for item in result["per_update"]
        ]
        args.json_output.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
