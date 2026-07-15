"""Pure (no-network) coverage for scripts/pack_hf_resource.py: normalization,
cache-folder / content-address naming, offline ref writing, and the sha-pinned
resolve. The huggingface_hub network paths run only in an end-to-end deploy."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import pytest

_PACKER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "pack_hf_resource.py"


def _load_packer():
    spec = importlib.util.spec_from_file_location("pack_hf_resource", _PACKER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


phr = _load_packer()


# --- normalize_resource ----------------------------------------------------


def test_normalize_string_shorthand():
    r = phr.normalize_resource("meta-llama/Llama-3.1-8B")
    assert r["repo_id"] == "meta-llama/Llama-3.1-8B"
    assert r["repo_type"] == "model"
    assert r["revision"] == "main"
    assert r["allow_patterns"] is None and r["ignore_patterns"] is None


def test_normalize_object_full():
    r = phr.normalize_resource(
        {
            "repo_id": "org/name",
            "revision": "v2",
            "repo_type": "dataset",
            "allow_patterns": ["*.json"],
            "ignore_patterns": ["*.bin"],
        }
    )
    assert r["revision"] == "v2" and r["repo_type"] == "dataset"
    assert r["allow_patterns"] == ["*.json"] and r["ignore_patterns"] == ["*.bin"]


@pytest.mark.parametrize(
    "entry",
    [
        {"repo_id": ""},
        {"repo_id": "a/b/c"},
        {"repo_id": "x", "repo_type": "space"},
        {"repo_id": "x", "junk": 1},
        {"repo_id": "x", "allow_patterns": [1]},
        5,
    ],
)
def test_normalize_rejects_bad(entry):
    with pytest.raises(ValueError):
        phr.normalize_resource(entry)


# --- names -----------------------------------------------------------------


def test_repo_folder_name():
    assert (
        phr.repo_folder_name("meta-llama/Llama-3.1-8B", "model")
        == "models--meta-llama--Llama-3.1-8B"
    )
    assert phr.repo_folder_name("bert-base-uncased", "model") == "models--bert-base-uncased"
    assert phr.repo_folder_name("org/ds", "dataset") == "datasets--org--ds"


def test_derive_names_full_repo_clean_sha_name():
    res = phr.normalize_resource("org/name")
    sha = "a" * 40
    n = phr.derive_names(res, sha)
    assert n["mount_local_path"] == "/tmp/.cache/huggingface/hub/models--org--name"
    assert n["remote_dir"] == "cache/huggingface/models--org--name"
    assert n["remote_basename"] == f"{sha}.squashfs"
    assert n["remote_name"].endswith(f"models--org--name/{sha}.squashfs")


def test_derive_names_patterned_gets_suffix():
    res = phr.normalize_resource({"repo_id": "org/name", "allow_patterns": ["*.safetensors"]})
    sha = "b" * 40
    base = phr.derive_names(res, sha)["remote_basename"]
    assert base.startswith(sha + "-") and base.endswith(".squashfs")


def test_pattern_suffix_stable_and_distinct():
    s1 = phr.pattern_suffix(["*.safetensors", "*.json"], None)
    s2 = phr.pattern_suffix(["*.json", "*.safetensors"], None)  # order-insensitive
    s3 = phr.pattern_suffix(["*.bin"], None)
    assert s1 == s2 and s1 != s3
    assert phr.pattern_suffix(None, None) == ""


# --- offline ref writing + sha-pinned resolve ------------------------------


def test_ensure_ref_writes_and_idempotent(tmp_path: Path):
    sha = "a" * 40
    phr._ensure_ref(tmp_path, "main", sha)
    assert (tmp_path / "refs" / "main").read_text() == sha
    phr._ensure_ref(tmp_path, "main", sha)  # idempotent
    assert (tmp_path / "refs" / "main").read_text() == sha


def test_ensure_ref_nested_slash_creates_parents(tmp_path: Path):
    sha = "b" * 40
    phr._ensure_ref(tmp_path, "release/v2", sha)  # branch name with a slash
    assert (tmp_path / "refs" / "release" / "v2").read_text() == sha


def test_refs_for_revision():
    assert phr._refs_for_revision("main") == ["main"]
    assert phr._refs_for_revision("v1.0") == ["main", "v1.0"]
    assert phr._refs_for_revision("release/v2") == ["main", "release/v2"]
    assert phr._refs_for_revision("a" * 40) == ["main"]  # bare sha needs no ref
    assert phr._refs_for_revision("A" * 40) == ["main"]  # case-insensitive


def test_resolve_sha_pinned_is_offline():
    # A sha-pinned revision resolves with NO network/token/huggingface_hub.
    out = phr.resolve(phr.normalize_resource({"repo_id": "org/name", "revision": "C" * 40}))
    assert out["commit_sha"] == "c" * 40
    assert out["remote_name"].endswith("/" + "c" * 40 + ".squashfs")


def test_load_resources_rejects_duplicate_repo():
    with tempfile.TemporaryDirectory() as d:
        mp = Path(d) / "metadata.json"
        mp.write_text(
            json.dumps(
                {"hf_resources": ["org/name", {"repo_id": "org/name", "allow_patterns": ["*.json"]}]}
            )
        )
        with pytest.raises(SystemExit):
            phr._load_resources_from_metadata(mp)


def test_load_resources_allows_same_name_different_type():
    with tempfile.TemporaryDirectory() as d:
        mp = Path(d) / "metadata.json"
        mp.write_text(
            json.dumps(
                {
                    "hf_resources": [
                        {"repo_id": "org/name", "repo_type": "model"},
                        {"repo_id": "org/name", "repo_type": "dataset"},
                    ]
                }
            )
        )
        assert len(phr._load_resources_from_metadata(mp)) == 2
