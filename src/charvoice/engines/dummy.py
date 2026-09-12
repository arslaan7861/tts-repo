"""A real, audible, deterministic CPU engine with no TTS model behind it.

Lets the whole pipeline (parsing, profiles, caching, audio assembly) be
exercised and tested with no GPU and no torch, per requirements.md section 2.
`generate_one` is deterministic by design -- same text + profile always
produces byte-identical samples -- because other packages' cache tests rely
on that.
"""

from __future__ import annotations

import hashlib

import numpy as np

from charvoice.audio import AudioSegment
from charvoice.config import EngineConfig
from charvoice.engines.base import Engine, SpeakerHandle
from charvoice.engines.registry import register
from charvoice.profiles import VoiceProfile

_DEFAULT_SAMPLE_RATE = 22050
_MIN_FREQ_HZ = 150.0
_MAX_FREQ_HZ = 400.0
_WORDS_PER_SECOND = 2.5
_MIN_DURATION_S = 0.3
_ENVELOPE_S = 0.015  # linear attack/decay, so concatenated segments don't click
_PEAK_AMPLITUDE = 0.3


def _stable_hash_int(text: str) -> int:
    """Deterministic integer hash, independent of process/PYTHONHASHSEED.

    Python's builtin hash() is salted per-process, which would make
    "same text+profile => byte-identical output" fail across runs.
    """
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def _frequency_for_speaker(speaker_id: str) -> float:
    """Map a speaker id to a fixed frequency in [_MIN_FREQ_HZ, _MAX_FREQ_HZ]."""
    fraction = (_stable_hash_int(speaker_id) % 10_000) / 10_000.0
    return _MIN_FREQ_HZ + fraction * (_MAX_FREQ_HZ - _MIN_FREQ_HZ)


def _duration_for_text(text: str, speed: float) -> float:
    """Word-count-proportional duration, adjusted for speaking speed."""
    words = max(len(text.split()), 1)
    duration = words / _WORDS_PER_SECOND
    duration = max(duration, _MIN_DURATION_S)
    return duration / speed if speed else duration


def _sine_tone(frequency: float, duration_s: float, sample_rate: int) -> np.ndarray:
    """A sine tone with a short linear attack/decay envelope, peak-scaled."""
    n = max(round(duration_s * sample_rate), 1)
    t = np.arange(n, dtype=np.float64) / sample_rate
    tone = np.sin(2.0 * np.pi * frequency * t)

    envelope = np.ones(n, dtype=np.float64)
    ramp_n = min(round(_ENVELOPE_S * sample_rate), n // 2)
    if ramp_n > 0:
        ramp = np.linspace(0.0, 1.0, ramp_n)
        envelope[:ramp_n] = ramp
        envelope[n - ramp_n :] = ramp[::-1]

    samples = tone * envelope * _PEAK_AMPLITUDE
    return samples.astype(np.float32)


@register("dummy")
class DummyEngine(Engine):
    """Deterministic sine-tone synthesizer. No model, no GPU, no network."""

    def __init__(self) -> None:
        self._sample_rate = _DEFAULT_SAMPLE_RATE

    def load(self, engine_config: EngineConfig) -> None:
        """No-op load; only picks up an optional sample_rate override."""
        self._sample_rate = int(engine_config.extra.get("sample_rate", _DEFAULT_SAMPLE_RATE))

    def prepare_speaker(self, profile: VoiceProfile) -> SpeakerHandle:
        return SpeakerHandle(profile=profile, engine_state=None)

    def generate_one(self, text: str, speaker: SpeakerHandle) -> AudioSegment:
        frequency = _frequency_for_speaker(speaker.profile.id)
        duration_s = _duration_for_text(text, speaker.profile.speed)
        samples = _sine_tone(frequency, duration_s, self._sample_rate)
        return AudioSegment(samples=samples, sample_rate=self._sample_rate)
