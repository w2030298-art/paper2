"""Tests for game-aware BaseTrainer hooks."""

from typing import Any, Dict

from src.trainer.base_trainer import BaseTrainer


class _Agent:
    game_aware_enabled = True


class _Trainer(BaseTrainer):
    def _collect_rollout(self) -> Dict[str, Any]:
        return {}

    def _update_step(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        return {"constraint_residual": 2.0, "dual_variable_mean": 1.0}


def test_base_trainer_game_aware_hooks(tmp_path) -> None:
    trainer = _Trainer(env=object(), agent=_Agent(), total_timesteps=1, save_dir=str(tmp_path))

    batch = trainer._build_game_aware_batch({"reward_components": {"delay_cost": 1.0}})
    metrics = trainer._apply_primal_dual_update({"constraint_residual": 2.0, "dual_variable_mean": 1.0})
    trainer._log_reward_breakdown({"delay_cost": 1.0})

    assert batch["game_aware"]["enabled"] is True
    assert metrics["game_aware/constraint_residual_mean"] > 0.0
    assert metrics["game_aware/dual/queue_stability"] > 0.0
    assert trainer.train_logs["reward_breakdown/delay_cost"] == [1.0]
