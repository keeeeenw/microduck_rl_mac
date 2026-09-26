import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_native_gpu_profile.py"
SPEC = importlib.util.spec_from_file_location("analyze_native_gpu_profile", SCRIPT)
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


def test_profile_summary_keeps_nested_timers_separate():
    records = [
        {
            "iteration": 1,
            "collection_seconds": 10.0,
            "learning_seconds": 1.0,
            "transitions_per_second": 100.0,
            "max_rss_bytes": 1000,
            "collision_narrowphase_seconds_total": 2.0,
            "collision_fallback_branch_seconds_total": 3.0,
            "collision_transfer_seconds_total": 0.5,
        },
        {
            "iteration": 2,
            "collection_seconds": 11.0,
            "learning_seconds": 1.0,
            "transitions_per_second": 90.0,
            "max_rss_bytes": 1200,
            "collision_narrowphase_seconds_total": 3.0,
            "collision_fallback_branch_seconds_total": 5.0,
            "collision_transfer_seconds_total": 1.0,
        },
    ]

    result = PROFILE.summarize(records, None, None)
    summary = result["summary"]
    assert summary["collection_plus_learning_seconds_median"] == 11.5
    assert summary["narrowphase_seconds_per_update_median"] == 1.0
    assert summary["fallback_branch_seconds_per_update_median"] == 2.0
    assert summary["collision_transfer_seconds_per_update_median"] == 0.5
    # These scopes overlap and are never collapsed into a fabricated total.
    assert "profiled_total_seconds" not in summary


def test_single_update_window_uses_nearest_predecessor():
    records = [{"iteration": i, "collision_evaluations_count": i * 100} for i in (1, 2, 3)]
    result = PROFILE.summarize(records, 3, 3)
    assert result["per_update"][0]["delta"]["collision_evaluations_count"] == 100


def test_missing_timing_fields_are_not_zipped_across_records():
    rows = [
        {"iteration": 1, "collection_seconds": 2},
        {"iteration": 2, "learning_seconds": 3},
        {"iteration": 3, "collection_seconds": 5, "learning_seconds": 7},
    ]
    assert PROFILE.summarize(rows, None, None)["summary"]["collection_plus_learning_seconds_median"] == 12


def test_counter_gaps_duplicates_and_resets_are_not_single_update_deltas():
    rows = [
        {"iteration": 1, "collision_evaluations_count": 10},
        {"iteration": 3, "collision_evaluations_count": 30},
        {"iteration": 3, "collision_evaluations_count": 31},
        {"iteration": 4, "collision_evaluations_count": 1},
        {"iteration": 5, "collision_evaluations_count": 6},
    ]
    result = PROFILE.summarize(rows, None, None)
    assert [item["delta"]["collision_evaluations_count"] for item in result["per_update"]] == [None, None, None, None, 5]


def test_jsonl_restart_preserves_file_order_and_resets_delta(tmp_path):
    import json
    path = tmp_path / "progress.jsonl"
    rows = [
        {"iteration": 1, "collision_evaluations_count": 100},
        {"iteration": 2, "collision_evaluations_count": 200},
        {"iteration": 1, "collision_evaluations_count": 2},
        {"iteration": 2, "collision_evaluations_count": 5},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = PROFILE.summarize(PROFILE.read_records(path), None, None)
    assert [item["delta"]["collision_evaluations_count"] for item in result["per_update"]] == [None, 100, None, 3]
