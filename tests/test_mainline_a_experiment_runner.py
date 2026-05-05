"""Tests for the mainline-A experiment runner."""

from argparse import Namespace

from scripts.run_mainline_a_experiments import resolve_plans


def test_runner_resolves_all_stage_plans() -> None:
    args = Namespace(
        config=None,
        dry_run=True,
        stage="all",
        resume=False,
        output_root="experiments/mainline_a",
        results_root="results/mainline_a",
    )

    plans = resolve_plans(args)

    assert [plan["stage"] for plan in plans] == ["N0", "N1", "N2", "N3"]

