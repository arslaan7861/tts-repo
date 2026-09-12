"""Tests for charvoice.profiles."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from charvoice.errors import ProfileLoadError, ProfileNotFoundError
from charvoice.profiles import (
    ProfileStore,
    fingerprint_profile,
    load_combined_file,
    load_profile_dir,
    load_profiles,
    merge_profiles,
    profile_from_dict,
)

try:
    import soundfile as sf

    HAVE_SOUNDFILE = True
except ImportError:  # pragma: no cover
    HAVE_SOUNDFILE = False


def _write_wav(path: Path, seconds: float = 0.1, freq: float = 220.0) -> None:
    """Write real (or fallback) audio bytes -- fingerprinting hashes raw bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if HAVE_SOUNDFILE:
        import numpy as np

        sr = 8000
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        data = (0.1 * np.sin(2 * np.pi * freq * t)).astype("float32")
        sf.write(str(path), data, sr)
    else:  # pragma: no cover
        path.write_bytes(b"RIFF....WAVEfmt arbitrary-bytes-for-testing")


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


# --------------------------------------------------------------------------
# load_profile_dir
# --------------------------------------------------------------------------


def test_load_profile_dir_only(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    _write_wav(voices / "miles" / "reference.wav")
    _write_yaml(
        voices / "miles" / "profile.yaml",
        {
            "engine": "gpt-sovits",
            "reference_audio": "reference.wav",
            "reference_text": "hello",
            "language": "en",
            "speed": 1.0,
        },
    )

    profiles = load_profile_dir(voices)
    assert set(profiles) == {"miles"}
    profile = profiles["miles"]
    assert profile.engine == "gpt-sovits"
    assert profile.reference_audio == (voices / "miles" / "reference.wav").resolve()
    assert profile.reference_audio.is_absolute()


def test_load_profile_dir_missing_returns_empty(tmp_path: Path) -> None:
    assert load_profile_dir(tmp_path / "does-not-exist") == {}


def test_load_profile_dir_relative_path_resolves_against_character_dir(tmp_path: Path) -> None:
    """voices/miles/profile.yaml with reference.wav -> voices/miles/reference.wav."""
    voices = tmp_path / "voices"
    _write_wav(voices / "miles" / "reference.wav")
    _write_yaml(
        voices / "miles" / "profile.yaml",
        {"engine": "dummy", "reference_audio": "reference.wav"},
    )

    profiles = load_profile_dir(voices)
    assert profiles["miles"].reference_audio == (voices / "miles" / "reference.wav").resolve()


def test_load_profile_dir_absolute_path_passthrough(tmp_path: Path) -> None:
    external = tmp_path / "elsewhere" / "ref.wav"
    _write_wav(external)
    voices = tmp_path / "voices"
    _write_yaml(
        voices / "miles" / "profile.yaml",
        {"engine": "dummy", "reference_audio": str(external)},
    )

    profiles = load_profile_dir(voices)
    assert profiles["miles"].reference_audio == external


# --------------------------------------------------------------------------
# load_combined_file
# --------------------------------------------------------------------------


def test_load_combined_file_only(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    _write_wav(voices / "miles" / "reference.wav")
    _write_yaml(
        voices / "characters.yaml",
        {
            "characters": {
                "miles": {
                    "engine": "gpt-sovits",
                    "reference_audio": "miles/reference.wav",
                    "language": "en",
                }
            }
        },
    )

    profiles = load_combined_file(voices / "characters.yaml")
    assert set(profiles) == {"miles"}
    assert profiles["miles"].reference_audio == (voices / "miles" / "reference.wav").resolve()


def test_load_combined_file_missing_returns_empty(tmp_path: Path) -> None:
    assert load_combined_file(tmp_path / "nope.yaml") == {}


def test_load_combined_file_relative_path_resolves_against_combined_dir(tmp_path: Path) -> None:
    """voices/characters.yaml with miles/reference.wav -> voices/miles/reference.wav."""
    voices = tmp_path / "voices"
    _write_wav(voices / "miles" / "reference.wav")
    _write_yaml(
        voices / "characters.yaml",
        {"characters": {"miles": {"engine": "dummy", "reference_audio": "miles/reference.wav"}}},
    )

    profiles = load_combined_file(voices / "characters.yaml")
    assert profiles["miles"].reference_audio == (voices / "miles" / "reference.wav").resolve()


def test_load_combined_file_malformed_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "characters.yaml"
    path.write_text("characters: [this, is, a, list, not, a, mapping]")
    with pytest.raises(ProfileLoadError, match=str(path)):
        load_combined_file(path)


def test_load_combined_file_non_mapping_top_level_raises(tmp_path: Path) -> None:
    path = tmp_path / "characters.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ProfileLoadError):
        load_combined_file(path)


# --------------------------------------------------------------------------
# merge + load_profiles precedence
# --------------------------------------------------------------------------


def test_merge_profiles_later_wins() -> None:
    a = profile_from_dict("x", {"engine": "a"}, Path("a.yaml"), Path("."))
    b = profile_from_dict("x", {"engine": "b"}, Path("b.yaml"), Path("."))
    merged = merge_profiles({"x": a}, {"x": b})
    assert merged["x"].engine == "b"


def test_load_profiles_directory_overrides_combined(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    _write_wav(voices / "miles" / "reference.wav")
    _write_wav(voices / "narrator" / "reference.wav")

    _write_yaml(
        voices / "characters.yaml",
        {
            "characters": {
                "miles": {
                    "engine": "gpt-sovits",
                    "reference_audio": "miles/reference.wav",
                    "speed": 1.0,
                },
                "narrator": {
                    "engine": "gpt-sovits",
                    "reference_audio": "narrator/reference.wav",
                    "speed": 1.0,
                },
            }
        },
    )
    # Per-character override for miles only.
    _write_yaml(
        voices / "miles" / "profile.yaml",
        {"engine": "gpt-sovits", "reference_audio": "reference.wav", "speed": 1.5},
    )

    store = load_profiles(voices, combined_file=voices / "characters.yaml")
    assert len(store) == 2
    assert store.get("miles").speed == 1.5  # dir override wins
    assert store.get("narrator").speed == 1.0  # combined entry survives


def test_load_profiles_both_missing_gives_empty_store(tmp_path: Path) -> None:
    store = load_profiles(tmp_path / "voices", combined_file=tmp_path / "characters.yaml")
    assert len(store) == 0
    assert store.ids() == []


# --------------------------------------------------------------------------
# ProfileStore
# --------------------------------------------------------------------------


def test_profile_store_get_unknown_raises_profile_not_found() -> None:
    store = ProfileStore({})
    with pytest.raises(ProfileNotFoundError):
        store.get("ghost")


def test_profile_store_contains_ids_values_sorted() -> None:
    a = profile_from_dict("b-char", {"engine": "dummy"}, Path("b.yaml"), Path("."))
    b = profile_from_dict("a-char", {"engine": "dummy"}, Path("a.yaml"), Path("."))
    store = ProfileStore({"b-char": a, "a-char": b})
    assert "b-char" in store
    assert "ghost" not in store
    assert len(store) == 2
    assert store.ids() == ["a-char", "b-char"]
    assert [p.id for p in store.values()] == ["a-char", "b-char"]


# --------------------------------------------------------------------------
# profile_from_dict error cases
# --------------------------------------------------------------------------


def test_profile_from_dict_missing_engine_raises() -> None:
    with pytest.raises(ProfileLoadError):
        profile_from_dict("x", {"reference_audio": "ref.wav"}, Path("p.yaml"), Path("."))


def test_profile_from_dict_non_mapping_raises() -> None:
    with pytest.raises(ProfileLoadError):
        profile_from_dict("x", ["not", "a", "mapping"], Path("p.yaml"), Path("."))


def test_profile_from_dict_extra_fields_passthrough() -> None:
    profile = profile_from_dict(
        "x",
        {"engine": "gpt-sovits", "top_k": 5, "style": "angry"},
        Path("p.yaml"),
        Path("."),
    )
    assert profile.extra == {"top_k": 5, "style": "angry"}


# --------------------------------------------------------------------------
# fingerprint_profile
# --------------------------------------------------------------------------


def test_fingerprint_changes_when_reference_audio_content_changes(tmp_path: Path) -> None:
    ref = tmp_path / "reference.wav"
    ref.write_bytes(b"original-audio-bytes-aaaaaaaaaa")
    profile = profile_from_dict(
        "x", {"engine": "dummy", "reference_audio": str(ref)}, Path("p.yaml"), tmp_path
    )
    fp1 = fingerprint_profile(profile)

    ref.write_bytes(b"different-audio-bytes-bbbbbbbbbb")
    fp2 = fingerprint_profile(profile)

    assert fp1 != fp2


def test_fingerprint_unchanged_when_file_moved_with_identical_content(tmp_path: Path) -> None:
    original = tmp_path / "a" / "reference.wav"
    _write_wav(original)

    profile_a = profile_from_dict(
        "x", {"engine": "dummy", "reference_audio": str(original)}, Path("p.yaml"), tmp_path
    )
    fp_a = fingerprint_profile(profile_a)

    moved = tmp_path / "b" / "reference.wav"
    moved.parent.mkdir(parents=True, exist_ok=True)
    moved.write_bytes(original.read_bytes())

    profile_b = profile_from_dict(
        "x", {"engine": "dummy", "reference_audio": str(moved)}, Path("p.yaml"), tmp_path
    )
    fp_b = fingerprint_profile(profile_b)

    assert fp_a == fp_b


def test_fingerprint_excludes_id_and_source_path(tmp_path: Path) -> None:
    ref = tmp_path / "reference.wav"
    _write_wav(ref)

    profile_1 = profile_from_dict(
        "character-one",
        {"engine": "dummy", "reference_audio": str(ref)},
        Path("one.yaml"),
        tmp_path,
    )
    profile_2 = profile_from_dict(
        "character-two",
        {"engine": "dummy", "reference_audio": str(ref)},
        Path("two.yaml"),
        tmp_path,
    )

    assert fingerprint_profile(profile_1) == fingerprint_profile(profile_2)


def test_fingerprint_changes_with_speed_or_pitch(tmp_path: Path) -> None:
    ref = tmp_path / "reference.wav"
    _write_wav(ref)

    base = profile_from_dict(
        "x",
        {"engine": "dummy", "reference_audio": str(ref), "speed": 1.0},
        Path("p.yaml"),
        tmp_path,
    )
    faster = profile_from_dict(
        "x",
        {"engine": "dummy", "reference_audio": str(ref), "speed": 1.2},
        Path("p.yaml"),
        tmp_path,
    )

    assert fingerprint_profile(base) != fingerprint_profile(faster)


def test_fingerprint_none_reference_audio_is_stable() -> None:
    profile = profile_from_dict("x", {"engine": "dummy"}, Path("p.yaml"), Path("."))
    assert fingerprint_profile(profile) == fingerprint_profile(profile)
