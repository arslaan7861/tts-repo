"""Configuration loading.

Every dataclass here is frozen, so pieces of the config can be safely embedded
in cache keys without worrying about later mutation.

Path handling rule (requirements.md section 2, and the Drive-mount trap): the
directories in `PathsConfig` are *relative* strings. They are resolved against
an explicit `project_dir` -- the local checkout locally, or the Google Drive
folder in Colab -- never against the process working directory. Resolve them
with `Config.resolve()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from charvoice.errors import ConfigError


@dataclass(frozen=True)
class AudioConfig:
    output_format: str = "wav"
    sample_rate: int = 48000
    channels: int = 1
    bit_depth: int = 16
    pause_between_lines_ms: int = 250
    pause_between_scenes_ms: int = 800
    # Applied after a narrator line instead of the default pause. None means
    # "use pause_between_lines_ms".
    pause_after_narrator_ms: int | None = None
    # None skips per-segment peak normalization entirely -- useful when the
    # engine already manages its own output level (e.g. F5-TTS's internal
    # target_rms) and independently re-scaling each line would fight it,
    # amplifying noise in quiet segments and making loudness jump line to
    # line. Prefer target_lufs (normalizes the whole assembled timeline
    # once) over this for final output leveling.
    peak_dbfs: float | None = -1.0
    # Loudness normalisation is opt-in; it needs the optional pyloudnorm dep.
    target_lufs: float | None = None

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ConfigError(f"audio.sample_rate must be positive, got {self.sample_rate}")
        if self.channels not in (1, 2):
            raise ConfigError(f"audio.channels must be 1 or 2, got {self.channels}")
        if self.bit_depth not in (16, 24, 32):
            raise ConfigError(f"audio.bit_depth must be 16, 24 or 32, got {self.bit_depth}")
        if self.output_format not in ("wav", "mp3"):
            raise ConfigError(f"audio.output_format must be wav or mp3, got {self.output_format!r}")
        for name in ("pause_between_lines_ms", "pause_between_scenes_ms"):
            if getattr(self, name) < 0:
                raise ConfigError(f"audio.{name} must not be negative")
        if self.pause_after_narrator_ms is not None and self.pause_after_narrator_ms < 0:
            raise ConfigError("audio.pause_after_narrator_ms must not be negative")

    @property
    def subtype(self) -> str:
        """libsndfile subtype for the configured bit depth."""
        return {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}[self.bit_depth]


@dataclass(frozen=True)
class PathsConfig:
    voices: str = "voices"
    # Optional single-file profile set, e.g. "voices/characters.yaml".
    voices_combined: str | None = None
    models: str = "models"
    scripts: str = "scripts"
    output: str = "output"
    temp: str = "tmp"
    cache: str = "tmp/cache"


@dataclass(frozen=True)
class EngineConfig:
    name: str
    model_dir: str | None = None
    # Engine-specific settings passed through untouched.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TTSConfig:
    default_engine: str = "dummy"
    engines: dict[str, EngineConfig] = field(default_factory=dict)

    def engine(self, name: str) -> EngineConfig:
        """Config for `name`, defaulting to a bare EngineConfig if unlisted."""
        return self.engines.get(name, EngineConfig(name=name))


@dataclass(frozen=True)
class Config:
    project_name: str = "character-voice-generator"
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    # Base directory every relative path in `paths` resolves against.
    project_dir: Path = field(default_factory=Path.cwd)

    def resolve(self, which: str) -> Path:
        """Absolute path for one of the `paths` entries, e.g. resolve("output")."""
        value = getattr(self.paths, which, None)
        if value is None:
            raise ConfigError(f"paths.{which} is not set")
        return resolve_path(self.project_dir, value)

    def with_project_dir(self, project_dir: Path | str) -> Config:
        """Re-base every relative path onto a new directory (used after mounting Drive)."""
        return replace(self, project_dir=Path(project_dir).expanduser().resolve())

    def with_default_engine(self, engine_name: str) -> Config:
        """Override `tts.default_engine` (e.g. a CLI --engine flag).

        Only changes the fallback used when a speaker's own profile doesn't
        name an engine -- a profile's explicit `engine:` still wins.
        """
        return replace(self, tts=replace(self.tts, default_engine=engine_name))


def resolve_path(base: Path | str, value: Path | str) -> Path:
    """Resolve `value` against `base`, leaving absolute values untouched."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(base).expanduser() / path).resolve()


def default_config() -> Config:
    """A usable config with no file on disk. Handy in tests."""
    return Config(tts=TTSConfig(default_engine="dummy", engines={"dummy": EngineConfig("dummy")}))


def _require_mapping(value: Any, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _known_fields(cls: type, data: dict[str, Any], where: str) -> dict[str, Any]:
    """Filter `data` to the dataclass's fields, rejecting unknown keys loudly.

    Silently ignoring a typo like `sample_rates:` would leave the user wondering
    why their setting had no effect.
    """
    allowed = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in {where}: {', '.join(unknown)}\n"
            f"valid keys: {', '.join(sorted(allowed))}"
        )
    return data


def _parse_tts(data: dict[str, Any]) -> TTSConfig:
    engines: dict[str, EngineConfig] = {}
    for name, raw in _require_mapping(data.get("engines"), "tts.engines").items():
        raw = _require_mapping(raw, f"tts.engines.{name}")
        known = {"model_dir"}
        engines[name] = EngineConfig(
            name=name,
            model_dir=raw.get("model_dir"),
            extra={k: v for k, v in raw.items() if k not in known},
        )
    default = data.get("default_engine", "dummy")
    return TTSConfig(default_engine=default, engines=engines)


def config_from_dict(data: dict[str, Any], project_dir: Path | str | None = None) -> Config:
    """Build a Config from an already-parsed mapping. Missing keys take defaults."""
    data = _require_mapping(data, "config")

    project = _require_mapping(data.get("project"), "project")
    audio_raw = _known_fields(AudioConfig, _require_mapping(data.get("audio"), "audio"), "audio")
    paths_raw = _known_fields(PathsConfig, _require_mapping(data.get("paths"), "paths"), "paths")

    try:
        audio = AudioConfig(**audio_raw)
        paths = PathsConfig(**paths_raw)
    except TypeError as exc:  # wrong value type for a known key
        raise ConfigError(str(exc)) from exc

    return Config(
        project_name=project.get("name", "character-voice-generator"),
        tts=_parse_tts(_require_mapping(data.get("tts"), "tts")),
        audio=audio,
        paths=paths,
        project_dir=Path(project_dir or Path.cwd()).expanduser().resolve(),
    )


def load_config(
    path: Path | str = "config.yaml",
    project_dir: Path | str | None = None,
    missing_ok: bool = True,
) -> Config:
    """Load config.yaml.

    `project_dir` defaults to the config file's own directory, which is what
    makes relative paths behave the same locally and on Drive. When the file is
    absent and `missing_ok`, defaults are returned.
    """
    config_path = Path(path).expanduser()
    if not config_path.exists():
        if missing_ok:
            parent = config_path.parent
            base = project_dir or (parent if str(parent) != "" else Path.cwd())
            return replace(default_config(), project_dir=Path(base).expanduser().resolve())
        raise ConfigError(f"config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {config_path}: {exc}") from exc

    return config_from_dict(raw, project_dir or config_path.resolve().parent)
