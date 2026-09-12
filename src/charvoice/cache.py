"""Segment cache: reuse a synthesized line across script revisions and sessions.

Cache keys change whenever text, speaker, engine, voice profile or model
configuration change (requirements.md section 20). The cache directory is
meant to live on Google Drive so it survives a Colab runtime restart
(requirements.md section 13/19), which drives two design choices here:

- Lookup is a single `Path.exists()` -- never a directory scan -- because
  Drive-backed filesystems make scans slow.
- Writes go through a temp-name-then-`os.replace` so a runtime that gets
  killed mid-write cannot leave a half-written file that a later run reads
  back as a cache hit.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from charvoice.config import EngineConfig


@dataclass(frozen=True)
class CacheKey:
    """Everything that determines whether a cached segment may be reused."""

    text: str
    speaker_id: str
    engine_name: str
    profile_fingerprint: str
    model_fingerprint: str

    def digest(self) -> str:
        """sha256 hex of this key's canonical JSON representation."""
        payload = {
            "text": self.text,
            "speaker_id": self.speaker_id,
            "engine_name": self.engine_name,
            "profile_fingerprint": self.profile_fingerprint,
            "model_fingerprint": self.model_fingerprint,
        }
        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SegmentCache:
    """Content-addressed store of synthesized audio segments, sharded on disk."""

    def __init__(self, cache_dir: Path, enabled: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def path_for(self, key: CacheKey) -> Path:
        """Canonical on-disk path for `key`, sharded by the first two digest chars.

        Sharding keeps any single directory from accumulating thousands of
        entries, which Drive-backed filesystems handle poorly.
        """
        digest = key.digest()
        return self.cache_dir / digest[:2] / f"{digest}.wav"

    def get(self, key: CacheKey) -> Path | None:
        """Return the cached audio path for `key`, or None on a miss.

        A zero-byte file counts as a miss: it can only be the leftover of an
        interrupted write (writes are atomic, so a complete file is never
        empty), and treating it as a hit would feed empty audio downstream.
        """
        if not self.enabled:
            return None

        path = self.path_for(key)
        if path.is_file() and path.stat().st_size > 0:
            self._hits += 1
            return path

        self._misses += 1
        return None

    def put(self, key: CacheKey, audio_path: Path) -> Path:
        """Copy `audio_path` into the cache under `key` and write a debug sidecar.

        The copy is atomic: written to a temp name in the same directory,
        then moved into place with `os.replace`, so a killed process never
        leaves a partially-written file at the final path.
        """
        dest = self.path_for(key)
        dest.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = dest.parent / f".{dest.name}.{uuid.uuid4().hex}.tmp"
        try:
            shutil.copyfile(audio_path, tmp_path)
            os.replace(tmp_path, dest)
        except OSError:
            tmp_path.unlink(missing_ok=True)
            raise

        sidecar = dest.with_suffix(dest.suffix + ".json")
        sidecar_tmp = dest.parent / f".{sidecar.name}.{uuid.uuid4().hex}.tmp"
        sidecar_tmp.write_text(
            json.dumps(
                {
                    "text": key.text,
                    "speaker_id": key.speaker_id,
                    "engine_name": key.engine_name,
                    "profile_fingerprint": key.profile_fingerprint,
                    "model_fingerprint": key.model_fingerprint,
                    "digest": key.digest(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        os.replace(sidecar_tmp, sidecar)

        return dest

    def clear(self) -> None:
        """Remove the entire cache directory and reset hit/miss counters."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self._hits = 0
        self._misses = 0


def fingerprint_engine_config(engine_config: EngineConfig) -> str:
    """sha256 hex fingerprint of the engine settings that affect synthesis output."""
    payload = {
        "name": engine_config.name,
        "model_dir": engine_config.model_dir,
        "extra": sorted(engine_config.extra.items()),
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
