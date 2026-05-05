"""Tests for optional dynamic-pricing env path."""

from src.environments.mec_v3.game_theory_env import GameTheoryMECEnv


def test_dynamic_pricing_metadata_when_enabled(tmp_path) -> None:
    cfg = tmp_path / "pricing.yaml"
    cfg.write_text(
        """
dynamic_pricing:
  enabled: true
  base_price: 1.0
  bounds: {min: 0.05, max: 10.0}
  components:
    queue: {enabled: true, alpha: 0.4}
    channel: {enabled: true, alpha: 0.2}
    migration: {enabled: true, alpha: 0.3}
""",
        encoding="utf-8",
    )
    env = GameTheoryMECEnv(
        num_agents=1,
        num_edge_servers=2,
        max_steps=1,
        enable_mainline_a=True,
        dynamic_pricing_config=str(cfg),
    )

    _, info = env.reset(seed=1)

    assert info["dynamic_price_metadata"]["enabled"] is True
    assert len(info["dynamic_price_metadata"]["prices"]) == 2

