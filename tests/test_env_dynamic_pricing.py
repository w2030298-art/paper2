"""Tests for optional dynamic-pricing env path."""

import numpy as np

from src.environments.mec_v3.game_theory_env import GameTheoryMECEnv


def _write_pricing_config(path, enabled: bool, queue_alpha: float) -> None:
    path.write_text(
        f"""
dynamic_pricing:
  enabled: {str(enabled).lower()}
  base_price: 1.0
  bounds: {{min: 0.05, max: 10.0}}
  components:
    queue: {{enabled: true, alpha: {queue_alpha}}}
    channel: {{enabled: true, alpha: 0.2}}
    migration: {{enabled: true, alpha: 0.3}}
""",
        encoding="utf-8",
    )


def test_dynamic_pricing_changes_step_price_and_reward_path(tmp_path) -> None:
    cfg = tmp_path / "pricing.yaml"
    static_cfg = tmp_path / "pricing_static.yaml"
    _write_pricing_config(cfg, enabled=True, queue_alpha=2.0)
    _write_pricing_config(static_cfg, enabled=False, queue_alpha=2.0)
    dynamic_env = GameTheoryMECEnv(
        num_agents=1,
        num_edge_servers=2,
        max_steps=1,
        enable_mainline_a=True,
        dynamic_pricing_config=str(cfg),
    )
    static_env = GameTheoryMECEnv(
        num_agents=1,
        num_edge_servers=2,
        max_steps=1,
        enable_mainline_a=True,
        dynamic_pricing_config=str(static_cfg),
    )
    action = [{"target": 1, "ratio": np.array([1.0, 1.0, 0.0], dtype=np.float32)}]

    dynamic_env.reset(seed=1)
    static_env.reset(seed=1)
    _, _, _, _, dynamic_info = dynamic_env.step(action)
    _, _, _, _, static_info = static_env.step(action)

    assert dynamic_info["dynamic_price_metadata"]["enabled"] is True
    assert len(dynamic_info["dynamic_price_metadata"]["prices"]) == 2
    assert dynamic_info["reward_terms"][0]["mainline_a_price_cost"] > 0.0
    assert dynamic_info["reward_terms"][0]["r_price"] < 0.0
    assert tuple(dynamic_info["dynamic_price_metadata"]["prices"]) != tuple(
        static_info["dynamic_price_metadata"]["prices"]
    )
