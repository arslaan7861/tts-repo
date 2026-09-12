"""GPT-SoVITS adapter -- the primary voice engine (requirements.md section 2).

This adapter expects a working GPT-SoVITS checkout/installation under the
configured `model_dir` (see requirements.md section 14 for model-management
conventions). It is structurally complete -- registration, config
resolution, the batching-by-speaker path, checkpoint-aware cache
fingerprinting -- but the actual inference calls are UNVERIFIED without a
GPU: this module is exercised locally only by a smoke test (registration and
pure-Python helpers), never real inference, until it is run on Colab.

Hard rule (requirements.md section 2): this module must be importable on a
machine with no torch and no GPU. `torch` (and anything that transitively
imports it, e.g. the actual GPT-SoVITS inference code) is imported only
inside method bodies that are actually invoked when this engine is used --
never at module scope. No network calls happen at import time either.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from charvoice.audio import AudioSegment
from charvoice.config import EngineConfig
from charvoice.engines.base import Engine, SpeakerHandle
from charvoice.engines.registry import register
from charvoice.errors import GenerationError, GPUUnavailableError, ModelNotFoundError
from charvoice.profiles import VoiceProfile

# Checkpoint files we expect to find under model_dir once it holds a real
# GPT-SoVITS install. Exact names follow the upstream repo's convention.
_EXPECTED_CHECKPOINTS = ("gpt.ckpt", "sovits.pth")


@register("gpt-sovits")
class GPTSoVITSEngine(Engine):
    """Adapter for the GPT-SoVITS voice-cloning TTS model.

    One instance = one loaded checkpoint, per the base `Engine` contract.
    `engine_state` on each `SpeakerHandle` holds the per-character reference
    features extracted once in `prepare_speaker`.
    """

    def __init__(self) -> None:
        self._model_dir: Path | None = None
        self._model: Any = None
        self._device: Any = None

    def required_profile_fields(self) -> set[str]:
        return {"reference_audio", "reference_text"}

    def _resolve_checkpoints(self, engine_config: EngineConfig) -> Path:
        """Validate model_dir and checkpoint files exist; return the resolved dir."""
        if not engine_config.model_dir:
            raise ModelNotFoundError(
                'gpt-sovits requires tts.engines["gpt-sovits"].model_dir to be set '
                "to a directory containing the GPT-SoVITS checkpoints."
            )

        model_dir = Path(engine_config.model_dir).expanduser()
        if not model_dir.is_dir():
            raise ModelNotFoundError(f"gpt-sovits model_dir does not exist: {model_dir}")

        missing = [name for name in _EXPECTED_CHECKPOINTS if not (model_dir / name).is_file()]
        if missing:
            raise ModelNotFoundError(
                f"gpt-sovits model_dir {model_dir} is missing checkpoint file(s): "
                f"{', '.join(missing)}"
            )
        return model_dir

    def load(self, engine_config: EngineConfig) -> None:
        """Resolve + validate the checkpoint directory, then load weights onto the GPU.

        Lazy torch import: this is the only place in the module that touches
        torch, and only when an actual gpt-sovits engine is being loaded.
        """
        model_dir = self._resolve_checkpoints(engine_config)

        import torch  # noqa: PLC0415 -- deliberately lazy, see module docstring

        if not torch.cuda.is_available():
            raise GPUUnavailableError(
                "gpt-sovits requires a CUDA GPU and none is available. "
                "Run this on a Colab GPU runtime."
            )

        self._model_dir = model_dir
        self._device = torch.device("cuda")
        # Real checkpoint loading (GPT + SoVITS weights, vocoder, etc.) goes
        # here once this is run against an actual GPT-SoVITS checkout.
        self._model = None

    def prepare_speaker(self, profile: VoiceProfile) -> SpeakerHandle:
        """Extract and cache reference-audio features for one character.

        Done once per character (requirements.md section 19), not once per
        line -- the extracted features are stashed in `engine_state` and
        reused by every subsequent generate call for this speaker.
        """
        # Real implementation would run the GPT-SoVITS reference encoder on
        # profile.reference_audio/profile.reference_text here and stash the
        # resulting tensors/features as engine_state.
        engine_state = {
            "reference_audio": profile.reference_audio,
            "reference_text": profile.reference_text,
        }
        return SpeakerHandle(profile=profile, engine_state=engine_state)

    def generate_batch(self, texts: list[str], speaker: SpeakerHandle) -> list[AudioSegment]:
        """Synthesize multiple same-speaker lines in one forward pass.

        GPT-SoVITS can batch multiple lines against a single cached reference
        (speaker.engine_state) more cheaply than calling generate_one in a
        loop -- this is the override requirements.md section 19 asks for.
        """
        if self._model_dir is None:
            raise GenerationError("gpt-sovits engine.load() must be called before generation.")

        # Real implementation would batch `texts` through the loaded model
        # against speaker.engine_state in a single forward pass.
        return [self._synthesize_line(text, speaker) for text in texts]

    def generate_one(self, text: str, speaker: SpeakerHandle) -> AudioSegment:
        """Single-line fallback; routes through the batched path."""
        return self.generate_batch([text], speaker)[0]

    def _synthesize_line(self, text: str, speaker: SpeakerHandle) -> AudioSegment:
        """Placeholder for the actual per-line inference call.

        Unverified without a GPU -- see module docstring. Raising here (rather
        than faking audio) keeps this path honest: calling it without a real
        model loaded is a bug, not a silently-wrong result.
        """
        raise GenerationError(
            "GPTSoVITSEngine inference is not yet implemented against a real "
            "GPT-SoVITS checkout. This adapter is structurally complete but "
            "unverified until run on a GPU (Colab)."
        )

    def unload(self) -> None:
        """Free GPU memory. Safe to call even if load() was never called."""
        self._model = None
        if self._device is not None:
            try:
                import torch  # noqa: PLC0415

                torch.cuda.empty_cache()
            except ImportError:
                pass
        self._device = None

    def model_fingerprint(self, engine_config: EngineConfig) -> str:
        """Fingerprint including checkpoint file content where resolvable.

        Falls back to the base (config-only) fingerprint when model_dir
        doesn't resolve yet -- e.g. before the files have been downloaded --
        so callers can still probe a cache key without raising.
        """
        if not engine_config.model_dir:
            return super().model_fingerprint(engine_config)

        model_dir = Path(engine_config.model_dir).expanduser()
        if not model_dir.is_dir():
            return super().model_fingerprint(engine_config)

        parts = [engine_config.name, str(model_dir)]
        for name in _EXPECTED_CHECKPOINTS:
            ckpt = model_dir / name
            if ckpt.is_file():
                stat = ckpt.stat()
                parts.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
            else:
                parts.append(f"{name}:missing")
        parts.append(str(sorted(engine_config.extra.items())))

        payload = "|".join(parts)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
