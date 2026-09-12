"""Engine interface: the seam between the pipeline and any specific TTS backend.

The pipeline must never depend on GPT-SoVITS (or any other backend) directly
-- see requirements.md section 15. Every adapter implements `Engine`; the
pipeline only ever talks to that interface.

One `Engine` instance = one loaded model, created once and held in the
registry's process-level cache (requirements.md section 19). Per-character
setup (e.g. encoding reference audio) is a separate, cheaper step captured in
`SpeakerHandle`, so it happens once per character rather than once per line.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from charvoice.audio import AudioSegment
from charvoice.config import EngineConfig
from charvoice.profiles import VoiceProfile


@dataclass(frozen=True)
class SpeakerHandle:
    """Opaque result of one-time per-character setup."""

    profile: VoiceProfile
    engine_state: Any = None


class Engine(ABC):
    """One instance = one loaded model. Created once, held in the registry cache."""

    @abstractmethod
    def load(self, engine_config: EngineConfig) -> None:
        """Load model weights/checkpoints. Called once."""

    @abstractmethod
    def prepare_speaker(self, profile: VoiceProfile) -> SpeakerHandle:
        """One-time per-character setup (e.g. encode reference audio)."""

    @abstractmethod
    def generate_one(self, text: str, speaker: SpeakerHandle) -> AudioSegment:
        """Synthesize exactly one line. Always required -- the fallback path."""

    def generate_batch(self, texts: list[str], speaker: SpeakerHandle) -> list[AudioSegment]:
        """Default: loop generate_one. Override for real batching/throughput."""
        return [self.generate_one(t, speaker) for t in texts]

    def required_profile_fields(self) -> set[str]:
        """Profile fields this engine needs populated, beyond the base ones.

        e.g. GPT-SoVITS needs {"reference_audio", "reference_text"}. Used by
        validation to produce a clean pre-flight error instead of a crash
        deep inside generate_one().
        """
        return set()

    def unload(self) -> None:  # noqa: B027 -- deliberate concrete no-op default
        """Free resources (e.g. GPU memory). Default no-op."""

    def supports_batching(self) -> bool:
        return type(self).generate_batch is not Engine.generate_batch

    def model_fingerprint(self, engine_config: EngineConfig) -> str:
        """Hash identifying the loaded model/checkpoint, for cache keys.

        Default: hash of the EngineConfig alone. gpt_sovits overrides this to
        also include checkpoint file content where resolvable.
        """
        extra = sorted(engine_config.extra.items())
        payload = f"{engine_config.name}|{engine_config.model_dir}|{extra}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
