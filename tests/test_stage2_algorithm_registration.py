"""Registration and config tests for Stage-2 CL-PPO/GAM-COMA algorithms."""

from pathlib import Path
from typing import Any

import yaml

from scripts.benchmark import (
    ALGORITHM_CLASSES,
    ALGO_ENV_MAP,
    DEEP_FUSION_ALGOS,
    MULTI_AGENT_ALGOS,
    ON_POLICY,
)


def _load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML config mapping."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_stage2_algorithms_are_registered_for_benchmark_selection() -> None:
    """Benchmark registries should expose the new Stage-2 algorithms."""
    assert ALGORITHM_CLASSES["CL-PPO"] == ("rl_algorithms.cl_ppo", "CLPPOAgent")
    assert ALGORITHM_CLASSES["GAM-COMA"] == ("rl_algorithms.gam_coma", "GAMCOMAAgent")
    assert ALGO_ENV_MAP["CL-PPO"] == "MEC-v1-game-theory-continuous-ma"
    assert ALGO_ENV_MAP["GAM-COMA"] == "MEC-v1-game-theory-discrete-ma"
    assert "CL-PPO" in ON_POLICY
    assert "GAM-COMA" in ON_POLICY
    assert "GAM-COMA" in MULTI_AGENT_ALGOS
    assert "GAM-COMA" in DEEP_FUSION_ALGOS


def test_cl_ppo_config_exposes_required_innovation_blocks() -> None:
    """CL-PPO config should be seeded from tuned PPO defaults with new blocks."""
    cfg = _load_yaml("configs/algorithms/cl_ppo.yaml")

    assert cfg["algorithm"]["name"] == "CL-PPO"
    assert cfg["algorithm"]["base_algorithm"] == "PPO"
    assert cfg["training"]["total_timesteps"] == 100000
    assert cfg["constraints"]["enabled"] is True
    assert cfg["risk"]["enabled"] is True
    assert cfg["safety_layer"]["enabled"] is True
    assert cfg["safety_layer"]["action_low"] == -1.0
    assert cfg["safety_layer"]["action_high"] == 1.0


def test_gam_coma_config_exposes_required_innovation_blocks() -> None:
    """GAM-COMA config should be seeded from tuned COMA defaults with graph/mask blocks."""
    cfg = _load_yaml("configs/algorithms/gam_coma.yaml")

    assert cfg["algorithm"]["name"] == "GAM-COMA"
    assert cfg["algorithm"]["base_algorithm"] == "COMA"
    assert cfg["training"]["total_timesteps"] == 100000
    assert cfg["graph_attention"]["enabled"] is True
    assert cfg["action_masking"]["enabled"] is True
    assert cfg["social_influence"]["enabled"] is False
    assert cfg["social_influence"]["coeff"] == 0.0
