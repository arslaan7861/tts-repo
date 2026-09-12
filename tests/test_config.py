from __future__ import annotations

from pathlib import Path

import pytest

from charvoice.config import (
    Config,
    EngineConfig,
    config_from_dict,
    default_config,
    load_config,
    resolve_path,
)
from charvoice.errors import ConfigError


def test_default_config_has_dummy_engine():
    cfg = default_config()
    assert cfg.tts.default_engine == "dummy"
    assert "dummy" in cfg.tts.engines


def test_load_config_missing_file_returns_defaults(tmp_path):
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg.audio.sample_rate == 48000


def test_load_config_missing_file_strict_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml", missing_ok=False)


def test_load_config_full_yaml(tmp_path):
    (tmp_path / "config.yaml").write_text(
        """
project:
  name: my-project
tts:
  default_engine: gpt-sovits
  engines:
    gpt-sovits:
      model_dir: models/gpt-sovits
audio:
  output_format: wav
  sample_rate: 44100
  channels: 2
  pause_between_lines_ms: 300
paths:
  voices: voices
  output: output
"""
    )
    cfg = load_config(tmp_path / "config.yaml")
    assert cfg.project_name == "my-project"
    assert cfg.tts.default_engine == "gpt-sovits"
    assert cfg.tts.engines["gpt-sovits"].model_dir == "models/gpt-sovits"
    assert cfg.audio.sample_rate == 44100
    assert cfg.audio.channels == 2
    assert cfg.audio.pause_between_lines_ms == 300


def test_load_config_partial_yaml_falls_back_to_defaults(tmp_path):
    (tmp_path / "config.yaml").write_text("audio:\n  sample_rate: 22050\n")
    cfg = load_config(tmp_path / "config.yaml")
    assert cfg.audio.sample_rate == 22050
    assert cfg.audio.channels == 1  # default untouched


def test_load_config_malformed_yaml_raises(tmp_path):
    (tmp_path / "config.yaml").write_text("audio:\n  sample_rate: [unclosed\n")
    with pytest.raises(ConfigError):
        load_config(tmp_path / "config.yaml")


def test_unknown_audio_key_raises():
    with pytest.raises(ConfigError):
        config_from_dict({"audio": {"sample_rates": 1}})


def test_unknown_paths_key_raises():
    with pytest.raises(ConfigError):
        config_from_dict({"paths": {"voice": "x"}})


def test_invalid_sample_rate_raises():
    with pytest.raises(ConfigError):
        config_from_dict({"audio": {"sample_rate": -1}})


def test_invalid_channels_raises():
    with pytest.raises(ConfigError):
        config_from_dict({"audio": {"channels": 3}})


def test_invalid_output_format_raises():
    with pytest.raises(ConfigError):
        config_from_dict({"audio": {"output_format": "flac"}})


def test_engine_config_extra_passthrough():
    cfg = config_from_dict(
        {
            "tts": {
                "default_engine": "gpt-sovits",
                "engines": {"gpt-sovits": {"model_dir": "models/x", "some_param": 1.5}},
            }
        }
    )
    engine_cfg = cfg.tts.engine("gpt-sovits")
    assert engine_cfg.model_dir == "models/x"
    assert engine_cfg.extra == {"some_param": 1.5}


def test_engine_defaults_for_unlisted_name():
    cfg = default_config()
    engine_cfg = cfg.tts.engine("some-new-engine")
    assert engine_cfg.name == "some-new-engine"
    assert engine_cfg.model_dir is None


def test_resolve_path_relative_against_base():
    result = resolve_path("/base/dir", "voices")
    assert result == Path("/base/dir/voices").resolve()


def test_resolve_path_absolute_passthrough():
    result = resolve_path("/base/dir", "/abs/path")
    assert result == Path("/abs/path")


def test_config_resolve_uses_project_dir(tmp_path):
    cfg = Config(project_dir=tmp_path)
    assert cfg.resolve("voices") == (tmp_path / "voices").resolve()


def test_with_project_dir_rebase(tmp_path):
    cfg = default_config()
    other = tmp_path / "drive_folder"
    other.mkdir()
    rebased = cfg.with_project_dir(other)
    assert rebased.resolve("output") == (other / "output").resolve()


def test_config_is_frozen():
    cfg = default_config()
    with pytest.raises(AttributeError):
        cfg.project_name = "changed"  # type: ignore[misc]


def test_load_config_project_dir_defaults_to_config_file_parent(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "config.yaml").write_text("paths:\n  voices: voices\n")
    cfg = load_config(sub / "config.yaml")
    assert cfg.project_dir == sub.resolve()
    assert cfg.resolve("voices") == (sub / "voices").resolve()


def test_engine_config_repr_is_hashable_friendly():
    ec = EngineConfig(name="dummy")
    # frozen dataclasses with dict fields aren't hashable themselves, but
    # should at least support equality comparison without raising
    assert ec == EngineConfig(name="dummy")
