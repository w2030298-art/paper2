"""Static tests for COMA externalized search hyperparameters."""

from pathlib import Path

import yaml


def test_coma_yaml_exposes_stage1_update_hyperparameters() -> None:
    cfg = yaml.safe_load(Path("configs/algorithms/coma.yaml").read_text(encoding="utf-8"))
    ac = cfg["algorithm"]

    assert ac["policy_clip_low"] == 0.8
    assert ac["policy_clip_high"] == 1.2
    assert ac["grad_clip"] == 0.5
    assert ac["critic_loss_coeff"] == 0.5
    assert ac["entropy_coeff"] == 0.0


def test_coma_algorithm_uses_externalized_update_hyperparameters() -> None:
    source = Path("rl_algorithms/coma.py").read_text(encoding="utf-8")

    for name in (
        "policy_clip_low",
        "policy_clip_high",
        "grad_clip",
        "critic_loss_coeff",
        "entropy_coeff",
    ):
        assert name in source
    assert "torch.clamp(ratio, self.policy_clip_low, self.policy_clip_high)" in source
    assert "self.critic_loss_coeff * value_loss" in source
    assert "self.entropy_coeff * entropy" in source
    assert "self.grad_clip" in source
