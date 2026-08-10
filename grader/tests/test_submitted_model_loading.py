"""load_submitted_model: deserializes an agent model artifact in the
privilege-dropped worker, proxies ``.predict(...)``, and raises AgentFault on
agent-caused load failures."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest

from grading.faults import AgentFault
from grading.helpers import load_submitted_model, run_model_module  # noqa: F401

_MODEL_MODULE = (
    "class Doubler:\n"
    "    def predict(self, x):\n"
    "        return [v * 2 for v in x]\n"
)


def _write_pickled_model(tmp_path: Path, *, name: str = "model.pkl") -> Path:
    # Pickle BY REFERENCE (mymodel.Doubler), the common sklearn/pickle case.
    (tmp_path / "mymodel.py").write_text(_MODEL_MODULE)
    sys.path.insert(0, str(tmp_path))
    try:
        import mymodel  # noqa: PLC0415

        artifact = tmp_path / name
        with open(artifact, "wb") as fh:
            pickle.dump(mymodel.Doubler(), fh)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("mymodel", None)
    return artifact


def test_load_submitted_model_proxies_predict(tmp_path: Path):
    artifact = _write_pickled_model(tmp_path)
    model = load_submitted_model(artifact)
    try:
        assert model.predict([1, 2, 3]) == [2, 4, 6]
    finally:
        model.close()


def test_load_submitted_model_missing_artifact_is_agent_fault(tmp_path: Path):
    with pytest.raises(AgentFault, match="Missing submitted model"):
        load_submitted_model(tmp_path / "nope.pkl")


def test_load_submitted_model_non_regular_file_is_agent_fault(tmp_path: Path):
    d = tmp_path / "model.pkl"
    d.mkdir()  # a directory where a file is expected
    with pytest.raises(AgentFault, match="not a regular file"):
        load_submitted_model(d)


def test_load_submitted_model_oversized_is_agent_fault(tmp_path: Path):
    artifact = _write_pickled_model(tmp_path)
    with pytest.raises(AgentFault, match="over the"):
        load_submitted_model(artifact, max_bytes=1)


def test_load_submitted_model_bad_deserializer_is_value_error(tmp_path: Path):
    artifact = _write_pickled_model(tmp_path)
    with pytest.raises(ValueError, match="deserializer must be one of"):
        load_submitted_model(artifact, deserializer="torch")
