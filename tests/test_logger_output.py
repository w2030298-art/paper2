"""Minimal tests for log output."""

import json

from src.utils.logger import Logger


def test_logger_writes_hparams_and_metrics_json(tmp_path):
    logger = Logger(
        log_dir=str(tmp_path),
        experiment_name="log_smoke",
        use_tensorboard=False,
    )

    logger.log_params({"lr": 1e-3, "seed": 42})
    logger.log({"loss": 1.23, "reward": 4.56})
    logger.save_metrics()
    logger.close()

    exp_dir = tmp_path / "log_smoke"
    hparams_path = exp_dir / "hparams.json"
    metrics_path = exp_dir / "metrics.json"

    assert hparams_path.exists()
    assert metrics_path.exists()
    assert json.loads(hparams_path.read_text()) == {"lr": 1e-3, "seed": 42}
    assert json.loads(metrics_path.read_text()) == {"loss": [1.23], "reward": [4.56]}
