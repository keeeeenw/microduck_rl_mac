"""Summarize and compare matched PPO training runs (CPU vs Metal) at N=2048."""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def load_run(log_dir: Path) -> Dict[str, Any]:
    status_file = log_dir / "status.json"
    progress_file = log_dir / "progress.jsonl"
    manifest_file = log_dir / "manifest.json"

    assert status_file.exists(), f"Missing status.json in {log_dir}"
    assert progress_file.exists(), f"Missing progress.jsonl in {log_dir}"

    status = json.loads(status_file.read_text())
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}

    records = []
    with progress_file.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    return {
        "status": status,
        "manifest": manifest,
        "records": records,
    }


def analyze_matched_runs(cpu_dir: Path, metal_dir: Path, output_file: Path = None) -> Dict[str, Any]:
    cpu_run = load_run(cpu_dir)
    metal_run = load_run(metal_dir)

    num_iters = min(len(cpu_run["records"]), len(metal_run["records"]))
    assert num_iters > 0, "No completed iterations found"

    comparison_records = []
    for i in range(num_iters):
        c_rec = cpu_run["records"][i]
        m_rec = metal_run["records"][i]

        c_coll = c_rec["collection_seconds"]
        m_coll = m_rec["collection_seconds"]
        c_learn = c_rec["learning_seconds"]
        m_learn = m_rec["learning_seconds"]
        c_total = c_coll + c_learn
        m_total = m_coll + m_learn

        c_tr_s = c_rec["transitions_per_second"]
        m_tr_s = m_rec["transitions_per_second"]

        coll_speedup = c_coll / m_coll if m_coll > 0 else 0.0
        total_speedup = c_total / m_total if m_total > 0 else 0.0

        comparison_records.append({
            "iteration": i + 1,
            "cpu_collection_s": c_coll,
            "metal_collection_s": m_coll,
            "collection_speedup": coll_speedup,
            "cpu_learning_s": c_learn,
            "metal_learning_s": m_learn,
            "cpu_total_s": c_total,
            "metal_total_s": m_total,
            "total_speedup": total_speedup,
            "cpu_tr_s": c_tr_s,
            "metal_tr_s": m_tr_s,
            "cpu_max_rss_mb": c_rec.get("max_rss_bytes", 0) / (1024 * 1024),
            "metal_max_rss_mb": m_rec.get("max_rss_bytes", 0) / (1024 * 1024),
            "metal_collision_evals": m_rec.get("collision_evaluations_count", 0),
            "metal_narrowphase_s": m_rec.get("collision_narrowphase_seconds_total", 0.0),
            "metal_fallback_s": m_rec.get("collision_fallback_branch_seconds_total", 0.0),
            "metal_transfer_bytes": m_rec.get("collision_transfer_bytes_total", 0),
        })

    # Summary aggregations
    avg_cpu_coll = sum(r["cpu_collection_s"] for r in comparison_records) / num_iters
    avg_metal_coll = sum(r["metal_collection_s"] for r in comparison_records) / num_iters
    avg_coll_speedup = avg_cpu_coll / avg_metal_coll if avg_metal_coll > 0 else 0.0

    avg_cpu_total = sum(r["cpu_total_s"] for r in comparison_records) / num_iters
    avg_metal_total = sum(r["metal_total_s"] for r in comparison_records) / num_iters
    avg_total_speedup = avg_cpu_total / avg_metal_total if avg_metal_total > 0 else 0.0

    summary = {
        "num_envs": cpu_run["status"].get("num_envs", 2048),
        "iterations_evaluated": num_iters,
        "avg_cpu_collection_s": avg_cpu_coll,
        "avg_metal_collection_s": avg_metal_coll,
        "avg_collection_speedup": avg_coll_speedup,
        "avg_cpu_total_s": avg_cpu_total,
        "avg_metal_total_s": avg_metal_total,
        "avg_total_speedup": avg_total_speedup,
        "cpu_status": cpu_run["status"].get("status"),
        "metal_status": metal_run["status"].get("status"),
        "cpu_export_parity": cpu_run["status"].get("export_parity", False),
        "metal_export_parity": metal_run["status"].get("export_parity", False),
        "cpu_onnx_sha256": cpu_run["status"].get("onnx_sha256"),
        "metal_onnx_sha256": metal_run["status"].get("onnx_sha256"),
        "per_iteration": comparison_records,
    }

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(summary, indent=2))

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-dir", type=Path, required=True, help="CPU log directory")
    parser.add_argument("--metal-dir", type=Path, required=True, help="Metal log directory")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    summary = analyze_matched_runs(args.cpu_dir, args.metal_dir, args.output)

    print("\n" + "=" * 80)
    print(f"MATCHED PPO TRAINING BENCHMARK SUMMARY (N={summary['num_envs']}, {summary['iterations_evaluated']} iterations)")
    print("=" * 80)
    print(f"{'Iter':<5} | {'CPU Coll (s)':<13} | {'Metal Coll (s)':<15} | {'Coll Speedup':<13} | {'CPU Tot (s)':<12} | {'Metal Tot (s)':<14} | {'Tot Speedup':<12}")
    print("-" * 92)
    for r in summary["per_iteration"]:
        print(f"{r['iteration']:<5} | {r['cpu_collection_s']:<13.3f} | {r['metal_collection_s']:<15.3f} | {r['collection_speedup']:<13.2f}x | {r['cpu_total_s']:<12.3f} | {r['metal_total_s']:<14.3f} | {r['total_speedup']:<12.2f}x")
    print("-" * 92)
    print(f"Average Collection Speedup: {summary['avg_collection_speedup']:.2f}x ({summary['avg_cpu_collection_s']:.2f}s vs {summary['avg_metal_collection_s']:.2f}s)")
    print(f"Average Total Speedup:      {summary['avg_total_speedup']:.2f}x ({summary['avg_cpu_total_s']:.2f}s vs {summary['avg_metal_total_s']:.2f}s)")
    print(f"CPU Export Parity:   {summary['cpu_export_parity']} (ONNX: {summary['cpu_onnx_sha256'][:16]}...)")
    print(f"Metal Export Parity: {summary['metal_export_parity']} (ONNX: {summary['metal_onnx_sha256'][:16]}...)")
    print("=" * 80)


if __name__ == "__main__":
    main()
