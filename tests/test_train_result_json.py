"""Tests for train.py result JSON helpers."""

from importlib import util
from pathlib import Path

import numpy as np


def _load_train_module():
    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / "scripts" / "train.py"
    spec = util.spec_from_file_location("scripts_train_module", train_path)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_to_jsonable_converts_numpy_float32_to_python_float() -> None:
    train_module = _load_train_module()
    converted = train_module._to_jsonable({"x": np.float32(1.5)})
    assert isinstance(converted["x"], float)
    assert converted["x"] == 1.5


def test_write_result_json_atomic(tmp_path) -> None:
    train_module = _load_train_module()
    output = tmp_path / "results" / "result.json"
    payload = {"message": "中文", "score": np.float32(2.5)}
    train_module._write_result_json(output, payload)

    import json

    with output.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    assert loaded["message"] == "中文"
    assert loaded["score"] == 2.5
