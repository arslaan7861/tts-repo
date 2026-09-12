"""GPT-SoVITS adapter -- the primary voice engine (requirements.md section 2).

Drives the real upstream `TTS` class from
https://github.com/RVC-Boss/GPT-SoVITS (GPT_SoVITS/TTS_infer_pack/TTS.py) in
process, rather than shelling out to its HTTP server (api_v2.py) -- an
in-process call avoids a second process to manage and lets `unload()` free
GPU memory directly.

Model files (v2 pretrained base, few-shot voice cloning, no training
required) come from https://huggingface.co/lj1995/GPT-SoVITS:

    model_dir/
    ├── gsv-v2final-pretrained/
    │   ├── s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt   (t2s / GPT weights)
    │   └── s2G2333k.pth                                       (vits / SoVITS weights)
    ├── chinese-hubert-base/      (cnhuhbert_base_path, despite the name -- it's
    │                              the feature extractor TTS.py always loads)
    └── chinese-roberta-wwm-ext-large/   (bert_base_path)

This adapter is structurally complete and calls the real inference API, but
is UNVERIFIED without a GPU -- exercised locally only by a smoke test
(registration, checkpoint resolution, fingerprinting), never real inference,
until it is run on Colab.

Hard rule (requirements.md section 2): this module must be importable on a
machine with no torch and no GPU. `torch` and the GPT-SoVITS package itself
are imported only inside method bodies that are actually invoked when this
engine is used -- never at module scope. No network calls happen at import
time either.
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

# Filenames inside model_dir, from the v2 pretrained release on HuggingFace
# (lj1995/GPT-SoVITS). Override any of these via tts.engines["gpt-sovits"].extra
# if you use a different version (v3/v4/v2Pro) or a fine-tuned checkpoint.
_DEFAULT_T2S_WEIGHTS = "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
_DEFAULT_VITS_WEIGHTS = "gsv-v2final-pretrained/s2G2333k.pth"
_DEFAULT_BERT_DIR = "chinese-roberta-wwm-ext-large"
_DEFAULT_CNHUBERT_DIR = "chinese-hubert-base"
_DEFAULT_VERSION = "v2"

# text_lang/prompt_lang values the v2 text cleaner accepts
# (GPT_SoVITS/text/cleaner.py's language_module_map).
_SUPPORTED_LANGUAGES = {"zh", "ja", "en", "ko", "yue"}


@register("gpt-sovits")
class GPTSoVITSEngine(Engine):
    """Adapter for the GPT-SoVITS voice-cloning TTS model.

    One instance = one loaded `TTS` object, per the base `Engine` contract.
    `engine_state` on each `SpeakerHandle` holds that character's reference
    audio path/text -- GPT-SoVITS re-encodes the reference per request
    rather than exposing a separate "extract once" step, so there is nothing
    heavier to cache here; the one-time cost this still avoids is re-loading
    the GPT/SoVITS checkpoints, which `load()` does once per engine instance.
    """

    def __init__(self) -> None:
        self._tts: Any = None
        self._model_dir: Path | None = None

    def required_profile_fields(self) -> set[str]:
        return {"reference_audio", "reference_text"}

    def _checkpoint_paths(self, engine_config: EngineConfig) -> dict[str, Path]:
        """Resolve every required checkpoint/model path under model_dir.

        Individual filenames can be overridden via `extra` for a different
        GPT-SoVITS version or a fine-tuned checkpoint.
        """
        if not engine_config.model_dir:
            raise ModelNotFoundError(
                'gpt-sovits requires tts.engines["gpt-sovits"].model_dir to be set '
                "to a directory holding the GPT-SoVITS pretrained models "
                "(see this module's docstring for the expected layout)."
            )

        model_dir = Path(engine_config.model_dir).expanduser()
        if not model_dir.is_dir():
            raise ModelNotFoundError(f"gpt-sovits model_dir does not exist: {model_dir}")

        extra = engine_config.extra
        paths = {
            "t2s_weights_path": model_dir / extra.get("t2s_weights", _DEFAULT_T2S_WEIGHTS),
            "vits_weights_path": model_dir / extra.get("vits_weights", _DEFAULT_VITS_WEIGHTS),
            "bert_base_path": model_dir / extra.get("bert_dir", _DEFAULT_BERT_DIR),
            "cnhuhbert_base_path": model_dir / extra.get("cnhuhbert_dir", _DEFAULT_CNHUBERT_DIR),
        }
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise ModelNotFoundError(
                f"gpt-sovits model_dir {model_dir} is missing expected file(s):\n  "
                + "\n  ".join(missing)
                + "\nDownload the pretrained models from "
                "https://huggingface.co/lj1995/GPT-SoVITS into model_dir."
            )
        return paths

    def load(self, engine_config: EngineConfig) -> None:
        """Resolve checkpoints and construct the upstream `TTS` object.

        Lazy imports: this is the only method that imports torch or the
        GPT-SoVITS package, and only when an actual gpt-sovits engine is
        being loaded -- see module docstring.
        """
        paths = self._checkpoint_paths(engine_config)

        import torch  # noqa: PLC0415 -- deliberately lazy, see module docstring

        if not torch.cuda.is_available():
            raise GPUUnavailableError(
                "gpt-sovits requires a CUDA GPU and none is available. "
                "Run this on a Colab GPU runtime."
            )

        import os

        # GPT-SoVITS's own package expects os.getcwd() to be its repo root at import time
        # (e.g. `now_dir = os.getcwd()` in TTS.py, `sys.path.append(f"{os.getcwd()}/...")` in sv.py).
        # We temporarily change cwd so its module-level path variables capture the correct location.
        model_dir = Path(engine_config.model_dir).expanduser()
        old_cwd = os.getcwd()
        os.chdir(model_dir)
        try:
            # Expected on sys.path because model_dir's repo root was cloned per requirements.md
            from GPT_SoVITS.TTS_infer_pack.TTS import TTS  # noqa: PLC0415

            version = engine_config.extra.get("version", _DEFAULT_VERSION)
            config = {
                "device": "cuda",
                "is_half": bool(engine_config.extra.get("is_half", True)),
                "version": version,
                "t2s_weights_path": str(paths["t2s_weights_path"]),
                "vits_weights_path": str(paths["vits_weights_path"]),
                "bert_base_path": str(paths["bert_base_path"]),
                "cnhuhbert_base_path": str(paths["cnhuhbert_base_path"]),
            }

            self._tts = TTS(config)
            self._model_dir = model_dir
        finally:
            os.chdir(old_cwd)

    def prepare_speaker(self, profile: VoiceProfile) -> SpeakerHandle:
        """Validate and stash this character's reference audio/text/language.

        GPT-SoVITS takes the reference per `run()` call rather than exposing
        a separate "encode once" step, so there is no heavier setup to do
        here -- this just fails early (before any generation) if a profile
        names a language the loaded version doesn't support.
        """
        language = profile.language or "en"
        if language not in _SUPPORTED_LANGUAGES:
            supported = ", ".join(sorted(_SUPPORTED_LANGUAGES))
            raise GenerationError(
                f'Voice profile "{profile.id}" has language "{language}", which '
                f"gpt-sovits does not support. Supported: {supported}."
            )

        engine_state = {
            "reference_audio": str(profile.reference_audio),
            "reference_text": profile.reference_text,
            "language": language,
        }
        return SpeakerHandle(profile=profile, engine_state=engine_state)

    def generate_one(self, text: str, speaker: SpeakerHandle) -> AudioSegment:
        """Synthesize exactly one line via the loaded TTS object.

        `generate_batch` is not overridden: GPT-SoVITS's own `run()` already
        takes one `text` per call (its `batch_size` parameter governs
        internal sentence-splitting, not multiple unrelated lines at once),
        so the base class's loop is the correct, not merely default,
        implementation here.
        """
        if self._tts is None:
            raise GenerationError("gpt-sovits engine.load() must be called before generation.")

        import numpy as np  # noqa: PLC0415
        import os

        state = speaker.engine_state
        request = {
            "text": text,
            "text_lang": state["language"],
            "ref_audio_path": state["reference_audio"],
            "prompt_text": state["reference_text"],
            "prompt_lang": state["language"],
            "speed_factor": speaker.profile.speed or 1.0,
        }

        old_cwd = os.getcwd()
        os.chdir(self._model_dir)
        try:
            last_sample_rate, last_chunk = None, None
            for sample_rate, chunk in self._tts.run(request):
                last_sample_rate, last_chunk = sample_rate, chunk
        except Exception as exc:  # noqa: BLE001 -- surface any engine failure uniformly
            raise GenerationError(
                f'gpt-sovits failed synthesizing line for "{speaker.profile.id}": {exc}'
            ) from exc
        finally:
            os.chdir(old_cwd)

        if last_chunk is None:
            raise GenerationError(f'gpt-sovits produced no audio for "{speaker.profile.id}".')

        # TTS.run() yields int16 PCM; AudioSegment's convention is float32 in [-1, 1].
        samples = last_chunk.astype(np.float32) / 32768.0
        return AudioSegment(samples=samples, sample_rate=last_sample_rate)

    def unload(self) -> None:
        """Free GPU memory. Safe to call even if load() was never called."""
        self._tts = None
        try:
            import torch  # noqa: PLC0415

            torch.cuda.empty_cache()
        except ImportError:
            pass

    def model_fingerprint(self, engine_config: EngineConfig) -> str:
        """Fingerprint including checkpoint file content where resolvable.

        Falls back to the base (config-only) fingerprint when model_dir
        doesn't resolve yet -- e.g. before the files have been downloaded --
        so callers can still probe a cache key without raising.
        """
        try:
            paths = self._checkpoint_paths(engine_config)
        except ModelNotFoundError:
            return super().model_fingerprint(engine_config)

        parts = [engine_config.name, engine_config.extra.get("version", _DEFAULT_VERSION)]
        for name, path in sorted(paths.items()):
            if path.is_file():
                stat = path.stat()
                parts.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
            else:
                # a directory (bert/cnhuhbert) -- hash its presence + name only,
                # walking every file inside would be needlessly slow per call
                parts.append(f"{name}:dir:{path.name}")
        parts.append(str(sorted(engine_config.extra.items())))

        payload = "|".join(parts)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
