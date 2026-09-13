"""F5-TTS adapter -- Flow Matching zero-shot voice cloning.

F5-TTS (https://github.com/SWivid/F5-TTS) uses Flow Matching rather than the
autoregressive/VITS approach of older models, which gives noticeably more
natural prosody on long inputs.

Quality tuning: every knob `F5TTS.infer()` accepts is read from
`EngineConfig.extra`, so `config.yaml` can change them with no code edit --
see `_TUNING_DEFAULTS`. The defaults here are deliberately quality-first
rather than F5's speed-first library defaults (notably `nfe_step`).

Note on licensing: F5-TTS *weights* are CC-BY-NC (non-commercial). The
adapter interface exists so the engine can be swapped if that becomes a
constraint.

Hard rule (requirements.md section 2): this module must be importable on a
machine with no torch and no GPU. `torch` and the F5-TTS package itself are
imported only inside method bodies that are actually invoked when this
engine is used -- never at module scope.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from charvoice.audio import AudioSegment
from charvoice.config import EngineConfig
from charvoice.engines.base import Engine, SpeakerHandle
from charvoice.engines.registry import register
from charvoice.errors import GenerationError, GPUUnavailableError
from charvoice.profiles import VoiceProfile

# Checkpoint variant. v1 is the one upstream flags as having "better training
# and inference performance"; "F5TTS_Base" and "E2TTS_Base" also exist.
_DEFAULT_MODEL = "F5TTS_v1_Base"

# Passed straight through to F5TTS.infer(). Keys match its parameter names so
# config.yaml reads the same as the upstream API docs.
#
# Where we deviate from F5's own defaults, and why:
#   nfe_step 32 -> 48  number of flow-matching solver steps. The single
#                      biggest quality knob; costs ~50% more GPU time per
#                      line. Use the notebook's A/B cell to pick your own.
#   remove_silence False -> True
#                      strips the model's leading/trailing silence, so the
#                      gap between lines is exactly our configured
#                      pause_between_lines_ms and nothing else.
# The rest keep F5's tuned values; they're exposed so they can be changed
# without touching code (e.g. raise cfg_strength if a voice drifts off its
# reference).
_TUNING_DEFAULTS: dict[str, Any] = {
    "nfe_step": 48,
    "cfg_strength": 2.0,
    "sway_sampling_coef": -1.0,
    "cross_fade_duration": 0.15,
    "target_rms": 0.1,
    "remove_silence": True,
}


def _tuning_from(engine_config: EngineConfig) -> dict[str, Any]:
    """Merge `EngineConfig.extra` over `_TUNING_DEFAULTS`.

    Only known tuning keys are picked up, so unrelated `extra` entries (a
    `model_dir`-adjacent setting, say) never leak into the infer() call as
    unexpected kwargs.
    """
    extra = engine_config.extra
    return {key: extra.get(key, default) for key, default in _TUNING_DEFAULTS.items()}


def _model_variant(engine_config: EngineConfig) -> str:
    return str(engine_config.extra.get("model", _DEFAULT_MODEL))


def _seed_for(speaker_id: str, text: str) -> int:
    """A stable per-line seed.

    F5's own default (`seed=None`) picks randomly each call, which silently
    breaks the segment cache's premise that identical inputs produce
    identical audio -- a cache hit and a fresh generation would differ.
    Deriving the seed from speaker+text keeps every line reproducible
    without making all lines share one seed (which would correlate their
    sampling noise).
    """
    digest = hashlib.sha256(f"{speaker_id}\x00{text}".encode()).digest()
    # F5 passes this to torch.manual_seed; keep it in int32 range.
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


@register("f5-tts")
class F5TTSEngine(Engine):
    """Adapter for the F5-TTS Flow Matching model.

    One instance = one loaded checkpoint, per the base `Engine` contract.
    Tuning values are captured at `load()` so `generate_one` and
    `model_fingerprint` agree on exactly what produced a given segment.
    """

    def __init__(self) -> None:
        self._f5tts: Any = None
        self._tuning: dict[str, Any] = dict(_TUNING_DEFAULTS)

    def load(self, engine_config: EngineConfig) -> None:
        import torch  # noqa: PLC0415 -- deliberately lazy, see module docstring

        if not torch.cuda.is_available():
            raise GPUUnavailableError(
                "F5-TTS requires a CUDA GPU and none is available. "
                "Run this on a Colab GPU runtime."
            )

        self._tuning = _tuning_from(engine_config)

        try:
            from f5_tts.api import F5TTS  # noqa: PLC0415
        except ImportError as exc:
            raise GenerationError(
                "The f5-tts python package is not installed. "
                "Run `pip install f5-tts` or ensure the colab bootstrap installed it."
            ) from exc

        self._f5tts = F5TTS(model=_model_variant(engine_config))

    def required_profile_fields(self) -> set[str]:
        return {"reference_audio", "reference_text"}

    def model_fingerprint(self, engine_config: EngineConfig) -> str:
        """Hash the checkpoint variant AND every tuning value.

        This is what makes tuning take effect: the pipeline folds this into
        each segment's cache key, so changing `nfe_step` in config.yaml
        invalidates previously cached audio instead of silently serving the
        old render back and making the change look like a no-op.
        """
        payload = {
            "engine": "f5-tts",
            "model": _model_variant(engine_config),
            "tuning": _tuning_from(engine_config),
        }
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

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

        try:
            wav, sr, _ = self._f5tts.infer(
                ref_file=state["reference_audio"],
                ref_text=state["reference_text"],
                gen_text=text,
                speed=speaker.profile.speed or 1.0,
                seed=_seed_for(speaker.profile.id, text),
                **self._tuning,
            )
        except Exception as exc:
            raise GenerationError(
                f'F5-TTS failed synthesizing line for "{speaker.profile.id}": {exc}'
            ) from exc

        samples = np.asarray(wav, dtype=np.float32)
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
