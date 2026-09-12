"""Tests for charvoice.cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from charvoice.cache import CacheKey, SegmentCache, fingerprint_engine_config
from charvoice.config import EngineConfig

try:
    import soundfile as sf

    HAVE_SOUNDFILE = True
except ImportError:  # pragma: no cover
    HAVE_SOUNDFILE = False


def _write_wav(path: Path, seconds: float = 0.1, freq: float = 220.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if HAVE_SOUNDFILE:
        import numpy as np

        sr = 8000
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        data = (0.1 * np.sin(2 * np.pi * freq * t)).astype("float32")
        sf.write(str(path), data, sr)
    else:  # pragma: no cover
        path.write_bytes(b"RIFF....WAVEfmt arbitrary-bytes-for-testing")


def _key(**overrides) -> CacheKey:
    fields = dict(
        text="Something feels wrong.",
        speaker_id="spider-man",
        engine_name="gpt-sovits",
        profile_fingerprint="fp-profile",
        model_fingerprint="fp-model",
    )
    fields.update(overrides)
    return CacheKey(**fields)


# --------------------------------------------------------------------------
# CacheKey.digest
# --------------------------------------------------------------------------


def test_digest_stable_for_same_inputs() -> None:
    assert _key().digest() == _key().digest()


@pytest.mark.parametrize(
    "overrides",
    [
        {"text": "different text"},
        {"speaker_id": "miles"},
        {"engine_name": "dummy"},
        {"profile_fingerprint": "other-fp"},
        {"model_fingerprint": "other-model-fp"},
    ],
)
def test_digest_changes_when_any_field_changes(overrides: dict) -> None:
    assert _key().digest() != _key(**overrides).digest()


def test_digest_is_hex_sha256_length() -> None:
    digest = _key().digest()
    assert len(digest) == 64
    int(digest, 16)  # raises if not hex


# --------------------------------------------------------------------------
# SegmentCache
# --------------------------------------------------------------------------


def test_get_miss_then_put_then_hit(tmp_path: Path) -> None:
    cache = SegmentCache(tmp_path / "cache")
    key = _key()
    src = tmp_path / "segment.wav"
    _write_wav(src)

    assert cache.get(key) is None
    assert cache.misses == 1
    assert cache.hits == 0

    dest = cache.put(key, src)
    assert dest.is_file()

    hit = cache.get(key)
    assert hit == dest
    assert cache.hits == 1


def test_path_for_is_sharded_by_digest_prefix(tmp_path: Path) -> None:
    cache = SegmentCache(tmp_path / "cache")
    key = _key()
    path = cache.path_for(key)
    digest = key.digest()

    assert path == tmp_path / "cache" / digest[:2] / f"{digest}.wav"


def test_put_writes_json_sidecar_with_key_fields(tmp_path: Path) -> None:
    cache = SegmentCache(tmp_path / "cache")
    key = _key()
    src = tmp_path / "segment.wav"
    _write_wav(src)

    dest = cache.put(key, src)
    sidecar = dest.with_suffix(dest.suffix + ".json")
    assert sidecar.is_file()

    import json

    data = json.loads(sidecar.read_text())
    assert data["text"] == key.text
    assert data["speaker_id"] == key.speaker_id
    assert data["digest"] == key.digest()


def test_zero_byte_file_treated_as_miss(tmp_path: Path) -> None:
    cache = SegmentCache(tmp_path / "cache")
    key = _key()
    path = cache.path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()  # zero bytes, as if a write was interrupted

    assert cache.get(key) is None
    assert cache.misses == 1


def test_disabled_cache_always_misses(tmp_path: Path) -> None:
    cache = SegmentCache(tmp_path / "cache", enabled=False)
    key = _key()
    src = tmp_path / "segment.wav"
    _write_wav(src)

    # put() still writes physically (callers may disable only lookups)...
    cache.put(key, src)
    # ...but get() must still report a miss when disabled.
    assert cache.get(key) is None


def test_put_is_atomic_no_leftover_tmp_files(tmp_path: Path) -> None:
    cache = SegmentCache(tmp_path / "cache")
    key = _key()
    src = tmp_path / "segment.wav"
    _write_wav(src)

    dest = cache.put(key, src)
    leftovers = [p for p in dest.parent.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_clear_removes_cache_dir_and_resets_counters(tmp_path: Path) -> None:
    cache = SegmentCache(tmp_path / "cache")
    key = _key()
    src = tmp_path / "segment.wav"
    _write_wav(src)
    cache.put(key, src)
    cache.get(key)

    cache.clear()

    assert not (tmp_path / "cache").exists()
    assert cache.hits == 0
    assert cache.misses == 0
    assert cache.get(key) is None  # fresh miss after clear


def test_put_overwrites_existing_entry(tmp_path: Path) -> None:
    cache = SegmentCache(tmp_path / "cache")
    key = _key()
    src1 = tmp_path / "segment1.wav"
    src2 = tmp_path / "segment2.wav"
    _write_wav(src1, freq=220.0)
    _write_wav(src2, freq=880.0)

    dest1 = cache.put(key, src1)
    content1 = dest1.read_bytes()
    dest2 = cache.put(key, src2)
    content2 = dest2.read_bytes()

    assert dest1 == dest2
    assert content1 != content2


# --------------------------------------------------------------------------
# fingerprint_engine_config
# --------------------------------------------------------------------------


def test_fingerprint_engine_config_stable() -> None:
    cfg = EngineConfig(name="gpt-sovits", model_dir="models/gpt-sovits", extra={"top_k": 5})
    assert fingerprint_engine_config(cfg) == fingerprint_engine_config(cfg)


def test_fingerprint_engine_config_changes_with_model_dir() -> None:
    a = EngineConfig(name="gpt-sovits", model_dir="models/v1")
    b = EngineConfig(name="gpt-sovits", model_dir="models/v2")
    assert fingerprint_engine_config(a) != fingerprint_engine_config(b)


def test_fingerprint_engine_config_changes_with_extra() -> None:
    a = EngineConfig(name="gpt-sovits", extra={"top_k": 5})
    b = EngineConfig(name="gpt-sovits", extra={"top_k": 10})
    assert fingerprint_engine_config(a) != fingerprint_engine_config(b)
