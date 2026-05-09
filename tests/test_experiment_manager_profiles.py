"""Tests for experiment manager environment profile propagation."""

from src.experiment.local_index import LocalExperimentIndex
from src.experiment.manager import ExperimentManager
from src.experiment.process_runner import ProcessRunner
from src.experiment.registry import AlgorithmRegistry
from src.experiment.state_store import JsonStateStore


def _manager(tmp_path) -> ExperimentManager:
    store = JsonStateStore(root_dir=tmp_path / "experiments")
    return ExperimentManager(
        store=store,
        registry=AlgorithmRegistry(),
        runner=ProcessRunner(store=store),
        index=LocalExperimentIndex(db_path=tmp_path / "experiments/.index.sqlite3"),
    )


def test_create_experiment_records_mainline_a_profile(tmp_path) -> None:
    """Manifests and train specs should record the default Mainline-A profile."""
    manager = _manager(tmp_path)
    manifest = manager.create_experiment(
        run_id="profile_demo",
        name="Profile Demo",
        algorithms=["grpo"],
        timesteps=100,
        seed=42,
        device="cpu",
        eval_episodes=1,
    )
    assert manifest.metadata["environment_profile"] == "mainline-a"
    assert manifest.algorithms[0].extra_args == ["--environment-profile", "mainline-a"]


def test_create_experiment_records_legacy_profile_when_explicit(tmp_path) -> None:
    """Legacy profile should propagate only when explicitly requested."""
    manager = _manager(tmp_path)
    manifest = manager.create_experiment(
        run_id="legacy_demo",
        name="Legacy Demo",
        algorithms=["grpo"],
        timesteps=100,
        seed=42,
        device="cpu",
        eval_episodes=1,
        environment_profile="legacy",
    )
    assert manifest.metadata["environment_profile"] == "legacy"
    assert manifest.algorithms[0].extra_args == ["--environment-profile", "legacy"]
