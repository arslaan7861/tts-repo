"""F5-TTS adapter -- an advanced Flow Matching TTS engine.

F5-TTS (https://github.com/SWivid/F5-TTS) provides state-of-the-art zero-shot 
voice cloning with significantly fewer artifacts than older VITS-based models. 
It uses Flow Matching to produce highly natural speech that perfectly replicates 
the reference voice's timbre, expression, and pacing.

Hard rule (requirements.md section 2): this module must be importable on a
machine with no torch and no GPU. `torch` and the F5-TTS package itself
are imported only inside method bodies that are actually invoked when this
engine is used -- never at module scope.
"""

from __future__ import annotations

import hashlib
from typing import Any

from charvoice.audio import AudioSegment
from charvoice.config import EngineConfig
from charvoice.engines.base import Engine, SpeakerHandle
from charvoice.engines.registry import register
from charvoice.errors import GenerationError, GPUUnavailableError
from charvoice.profiles import VoiceProfile


@register("f5-tts")
class F5TTSEngine(Engine):
    def __init__(self) -> None:
        self._f5tts = None

    def load(self, engine_config: EngineConfig) -> None:
        import torch  # noqa: PLC0415
        
        if not torch.cuda.is_available():
            raise GPUUnavailableError(
                "F5-TTS requires a CUDA GPU and none is available. "
                "Run this on a Colab GPU runtime."
            )

        try:
            from f5_tts.api import F5TTS  # noqa: PLC0415
            self._f5tts = F5TTS()
        except ImportError as exc:
            raise GenerationError(
                "The f5-tts python package is not installed. "
                "Run `pip install f5-tts` or ensure the colab bootstrap installed it."
            ) from exc

    def required_profile_fields(self) -> set[str]:
        return {"reference_audio", "reference_text"}

    def model_fingerprint(self, engine_config: EngineConfig) -> str:
        # F5-TTS manages its own checkpoints via huggingface_hub. 
        # For now, it just uses the default base model so the fingerprint is static.
        return "f5-tts-base"

    def prepare_speaker(self, profile: VoiceProfile) -> SpeakerHandle:
        engine_state = {
            "reference_audio": str(profile.reference_audio.resolve()),
            "reference_text": profile.reference_text,
        }
        return SpeakerHandle(profile=profile, engine_state=engine_state)

    def generate_one(self, text: str, speaker: SpeakerHandle) -> AudioSegment:
        if self._f5tts is None:
            raise GenerationError("F5-TTS engine.load() must be called before generation.")

        import numpy as np  # noqa: PLC0415

        state = speaker.engine_state
        ref_audio = state["reference_audio"]
        ref_text = state["reference_text"]

        try:
            # F5TTS infer typically returns (wav_array, sample_rate, spectrogram)
            wav, sr, _ = self._f5tts.infer(
                ref_file=ref_audio,
                ref_text=ref_text,
                gen_text=text,
                speed=speaker.profile.speed or 1.0,
            )
        except Exception as exc:
            raise GenerationError(
                f'F5-TTS failed synthesizing line for "{speaker.profile.id}": {exc}'
            ) from exc

        # F5TTS returns float32 numpy array usually
        samples = np.array(wav, dtype=np.float32)
        # Ensure mono
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        return AudioSegment(samples=samples, sample_rate=sr)

    def unload(self) -> None:
        if self._f5tts is not None:
            import gc  # noqa: PLC0415
            import torch  # noqa: PLC0415
            del self._f5tts
            self._f5tts = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
