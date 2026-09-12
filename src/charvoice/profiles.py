"""Voice profile loading and fingerprinting.

A profile describes one character's voice configuration (requirements.md
section 4). Profiles can live in two places at once -- per-character files
under `voices/<id>/profile.yaml`, and/or a single combined file -- and both
are merged, with per-character files winning (requirements.md section 5).

Path handling rule, and the reason this module exists separately from
`config.py`: a profile's `reference_audio` resolves relative to the
directory of the YAML file that declared it, not the cwd and not
`paths.voices`. That is what lets the same profile set work unmodified
whether it sits in the local checkout or a mounted Google Drive folder.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from charvoice.config import resolve_path
from charvoice.errors import ProfileLoadError, ProfileNotFoundError

# Fields that change synthesized audio. Order matters only for readability;
# the canonical JSON payload sorts keys itself.
_FINGERPRINT_FIELDS = ("reference_text", "language", "speed", "pitch")


@dataclass(frozen=True)
class VoiceProfile:
    """One character's voice configuration.

    `reference_audio` is always an absolute, already-resolved path (or None).
    `source_path` records where the profile was loaded from, for error
    messages only -- it is excluded from equality so two profiles with the
    same settings loaded from different files still compare equal.
    """

    id: str
    engine: str
    reference_audio: Path | None = None
    reference_text: str | None = None
    language: str = "en"
    speed: float = 1.0
    pitch: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = field(default=None, compare=False)


class ProfileStore:
    """Lookup table of voice profiles keyed by character id."""

    def __init__(self, profiles: dict[str, VoiceProfile]) -> None:
        self._profiles = dict(profiles)

    def get(self, speaker_id: str) -> VoiceProfile:
        """Return the profile for `speaker_id`.

        Raises:
            ProfileNotFoundError: no profile is registered under that id.
        """
        try:
            return self._profiles[speaker_id]
        except KeyError:
            raise ProfileNotFoundError(
                f'Character "{speaker_id}" appears in the script but has no voice profile.\n\n'
                f"Create:\nvoices/{speaker_id}/profile.yaml"
            ) from None

    def __contains__(self, speaker_id: str) -> bool:
        return speaker_id in self._profiles

    def __len__(self) -> int:
        return len(self._profiles)

    def ids(self) -> list[str]:
        """Registered character ids, sorted."""
        return sorted(self._profiles)

    def values(self) -> list[VoiceProfile]:
        """All profiles, in id-sorted order."""
        return [self._profiles[i] for i in self.ids()]


def _require_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProfileLoadError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def profile_from_dict(
    profile_id: str,
    data: dict[str, Any],
    source_path: Path,
    base_dir: Path,
) -> VoiceProfile:
    """Build a VoiceProfile from a parsed mapping.

    `base_dir` is the directory `reference_audio` resolves against -- the
    directory containing the YAML file that declared this profile, per the
    module-level path rule.
    """
    data = _require_mapping(data, f"profile {profile_id!r} in {source_path}")

    known = {"engine", "reference_audio", "reference_text", "language", "speed", "pitch"}
    try:
        engine = data["engine"]
    except KeyError:
        raise ProfileLoadError(
            f"profile {profile_id!r} in {source_path} is missing required key: engine"
        ) from None

    reference_audio = data.get("reference_audio")
    resolved_audio = resolve_path(base_dir, reference_audio) if reference_audio else None

    try:
        return VoiceProfile(
            id=profile_id,
            engine=engine,
            reference_audio=resolved_audio,
            reference_text=data.get("reference_text"),
            language=data.get("language", "en"),
            speed=data.get("speed", 1.0),
            pitch=data.get("pitch"),
            extra={k: v for k, v in data.items() if k not in known},
            source_path=source_path,
        )
    except TypeError as exc:  # wrong value type for a known key
        raise ProfileLoadError(f"profile {profile_id!r} in {source_path}: {exc}") from exc


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ProfileLoadError(f"could not parse {path}: {exc}") from exc
    except OSError as exc:
        raise ProfileLoadError(f"could not read {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ProfileLoadError(f"{path} must contain a mapping, got {type(raw).__name__}")
    return raw


def load_profile_dir(voices_dir: Path) -> dict[str, VoiceProfile]:
    """Load every `voices/<id>/profile.yaml` under `voices_dir`.

    Missing `voices_dir` yields an empty dict, not an error -- validation
    catches an empty profile set later, once it knows whether the script
    actually needs any.
    """
    voices_dir = Path(voices_dir)
    if not voices_dir.is_dir():
        return {}

    profiles: dict[str, VoiceProfile] = {}
    for child in sorted(voices_dir.iterdir()):
        profile_path = child / "profile.yaml"
        if not child.is_dir() or not profile_path.is_file():
            continue
        data = _load_yaml_mapping(profile_path)
        profiles[child.name] = profile_from_dict(child.name, data, profile_path, base_dir=child)
    return profiles


def load_combined_file(path: Path) -> dict[str, VoiceProfile]:
    """Load the `characters:` mapping from a single combined profile file.

    Missing `path` yields an empty dict, not an error.
    """
    path = Path(path)
    if not path.is_file():
        return {}

    raw = _load_yaml_mapping(path)
    characters = raw.get("characters", {})
    characters = _require_mapping(characters, f"characters in {path}")

    base_dir = path.parent
    return {
        profile_id: profile_from_dict(profile_id, data, path, base_dir=base_dir)
        for profile_id, data in characters.items()
    }


def merge_profiles(*sources: dict[str, VoiceProfile]) -> dict[str, VoiceProfile]:
    """Merge profile dicts left to right; later sources override earlier ones."""
    merged: dict[str, VoiceProfile] = {}
    for source in sources:
        merged.update(source)
    return merged


def load_profiles(
    voices_dir: Path,
    combined_file: Path | None = None,
    project_dir: Path | None = None,
) -> ProfileStore:
    """Load and merge profiles from a per-character directory and/or a combined file.

    Directory profiles override combined-file entries with the same id: the
    combined file is the base carried in Drive, per-character files are the
    hand-edited override layer (requirements.md section 5). `project_dir` is
    accepted for symmetry with `Config` but is unused here -- both
    `voices_dir` and `combined_file` are expected to already be absolute.
    """
    del project_dir  # unused; paths are resolved relative to their own file

    dir_profiles = load_profile_dir(Path(voices_dir))
    combined_profiles = load_combined_file(Path(combined_file)) if combined_file else {}
    return ProfileStore(merge_profiles(combined_profiles, dir_profiles))


# --------------------------------------------------------------------------
# Fingerprinting
# --------------------------------------------------------------------------

# Keyed by (path, mtime_ns, size) -> content hash, so repeated fingerprinting
# of an unchanged file (e.g. once per script line) does not re-read it.
_content_hash_cache: dict[tuple[str, int, int], str] = {}


def _file_content_hash(path: Path) -> str:
    """sha256 hex of `path`'s bytes, cached by (path, mtime, size).

    We hash content, not the path, because the whole point of the fingerprint
    is to invalidate the synthesis cache when the *audio itself* changes --
    someone re-recording a reference clip at the same filename must bust the
    cache; moving the same file to a new directory must not.
    """
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _content_hash_cache.get(key)
    if cached is not None:
        return cached

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _content_hash_cache[key] = digest
    return digest


def fingerprint_profile(profile: VoiceProfile) -> str:
    """sha256 hex fingerprint of the fields that change synthesized audio.

    Deliberately excludes `id` and `source_path`: two differently-named or
    differently-located profiles with identical voice settings should
    synthesize identically and share a cache entry. `reference_audio` is
    represented by its file *content* hash, not its path, so edits to the
    referenced audio invalidate the cache even though the path is unchanged.
    """
    payload: dict[str, Any] = {name: getattr(profile, name) for name in _FINGERPRINT_FIELDS}
    payload["reference_audio_hash"] = (
        _file_content_hash(profile.reference_audio) if profile.reference_audio else None
    )
    payload["extra"] = sorted(profile.extra.items())

    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
