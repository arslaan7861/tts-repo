"""Tests for charvoice.engines: registry caching, base defaults, dummy engine.

VoiceProfile/AudioSegment are owned by other in-flight agents (profiles.py,
audio.py). We try the real import first; if those modules don't exist yet in
this worktree, we fall back to minimal local stand-ins with the same field
names/types, so these tests run standalone and self-heal once the real
modules land -- no edits needed here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    from charvoice.profiles import VoiceProfile
except ImportError:
    from dataclasses import dataclass, field
    from pathlib import Path
    from typing import Any

    @dataclass(frozen=True)
    class VoiceProfile:  # type: ignore[no-redef]
        id: str
        engine: str
        reference_audio: Path | None = None
        reference_text: str | None = None
        language: str = "en"
        speed: float = 1.0
        pitch: float | None = None
        extra: dict[str, Any] = field(default_factory=dict)
        source_path: Path | None = None

try:
    from charvoice.audio import AudioSegment
except ImportError:
    from dataclasses import dataclass as _dataclass

    @_dataclass(frozen=True)
    class AudioSegment:  # type: ignore[no-redef]
        samples: np.ndarray
        sample_rate: int

        @property
        def channels(self) -> int:
            return 1 if self.samples.ndim == 1 else self.samples.shape[1]

        @property
        def duration_s(self) -> float:
            return len(self.samples) / float(self.sample_rate)


from charvoice.config import Config, EngineConfig, TTSConfig
from charvoice.engines import registry
from charvoice.engines.base import Engine, SpeakerHandle
from charvoice.engines.dummy import DummyEngine
from charvoice.errors import EngineNotFoundError


def _config(model_dir: str | None = None) -> Config:
    return Config(
        tts=TTSConfig(
            default_engine="dummy",
            engines={"dummy": EngineConfig(name="dummy", model_dir=model_dir)},
        )
    )


def _profile(profile_id: str = "narrator", speed: float = 1.0) -> VoiceProfile:
    return VoiceProfile(id=profile_id, engine="dummy", speed=speed)


@pytest.fixture(autouse=True)
def _clear_cache():
    registry.clear_engine_cache()
    yield
    registry.clear_engine_cache()


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_get_engine_returns_dummy_instance():
    engine = registry.get_engine("dummy", _config())
    assert isinstance(engine, DummyEngine)


def test_get_engine_same_config_returns_same_instance():
    config = _config()
    first = registry.get_engine("dummy", config)
    second = registry.get_engine("dummy", config)
    assert first is second


def test_get_engine_different_config_returns_different_instance():
    first = registry.get_engine("dummy", _config(model_dir=None))
    second = registry.get_engine("dummy", _config(model_dir="/some/other/dir"))
    assert first is not second


def test_get_engine_unknown_name_raises_with_available_list():
    with pytest.raises(EngineNotFoundError) as exc_info:
        registry.get_engine("not-a-real-engine", _config())
    assert "dummy" in str(exc_info.value)


def test_available_engines_includes_dummy():
    names = registry.available_engines()
    assert "dummy" in names


def test_available_engines_includes_gpt_sovits_if_importable():
    try:
        import charvoice.engines.gpt_sovits  # noqa: F401
    except ImportError:
        pytest.skip("gpt_sovits's own optional deps are not installed here")
    names = registry.available_engines()
    assert "gpt-sovits" in names


# --------------------------------------------------------------------------
# Engine base defaults
# --------------------------------------------------------------------------


class _MinimalEngine(Engine):
    """Implements only the required methods, to test base-class defaults."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def load(self, engine_config: EngineConfig) -> None:
        pass

    def prepare_speaker(self, profile: VoiceProfile) -> SpeakerHandle:
        return SpeakerHandle(profile=profile)

    def generate_one(self, text: str, speaker: SpeakerHandle) -> AudioSegment:
        self.calls.append(text)
        return AudioSegment(samples=np.zeros(4, dtype=np.float32), sample_rate=100)


class _BatchingEngine(_MinimalEngine):
    """Overrides generate_batch, to test supports_batching() detection."""

    def generate_batch(self, texts: list[str], speaker: SpeakerHandle) -> list[AudioSegment]:
        return [self.generate_one(t, speaker) for t in texts]


def test_default_generate_batch_loops_generate_one():
    engine = _MinimalEngine()
    speaker = engine.prepare_speaker(_profile())
    results = engine.generate_batch(["one", "two", "three"], speaker)
    assert engine.calls == ["one", "two", "three"]
    assert len(results) == 3
    assert all(isinstance(r, AudioSegment) for r in results)


def test_supports_batching_false_for_plain_subclass():
    assert _MinimalEngine().supports_batching() is False


def test_supports_batching_true_when_overridden():
    assert _BatchingEngine().supports_batching() is True


# --------------------------------------------------------------------------
# DummyEngine
# --------------------------------------------------------------------------


def test_dummy_generate_one_is_deterministic():
    engine = DummyEngine()
    engine.load(EngineConfig(name="dummy"))
    speaker = engine.prepare_speaker(_profile("narrator"))

    first = engine.generate_one("Hello there.", speaker)
    second = engine.generate_one("Hello there.", speaker)

    assert first.sample_rate == second.sample_rate
    np.testing.assert_array_equal(first.samples, second.samples)


def test_dummy_different_speakers_produce_different_frequencies():
    engine = DummyEngine()
    engine.load(EngineConfig(name="dummy"))

    speaker_a = engine.prepare_speaker(_profile("narrator"))
    speaker_b = engine.prepare_speaker(_profile("spider-man"))

    seg_a = engine.generate_one("Same line of text.", speaker_a)
    seg_b = engine.generate_one("Same line of text.", speaker_b)

    assert seg_a.samples.shape == seg_b.samples.shape
    assert not np.array_equal(seg_a.samples, seg_b.samples)

    # Dominant frequency via FFT peak should differ between the two speakers.
    def dominant_freq(samples: np.ndarray, sample_rate: int) -> float:
        spectrum = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
        return float(freqs[np.argmax(spectrum)])

    freq_a = dominant_freq(seg_a.samples, seg_a.sample_rate)
    freq_b = dominant_freq(seg_b.samples, seg_b.sample_rate)
    assert freq_a != freq_b


def test_dummy_output_is_non_silent():
    engine = DummyEngine()
    engine.load(EngineConfig(name="dummy"))
    speaker = engine.prepare_speaker(_profile())

    seg = engine.generate_one("This line should produce audible sound.", speaker)
    rms = float(np.sqrt(np.mean(np.square(seg.samples))))
    assert rms > 0.01
    assert seg.samples.dtype == np.float32
    assert seg.samples.ndim == 1


def test_dummy_longer_text_produces_longer_duration():
    engine = DummyEngine()
    engine.load(EngineConfig(name="dummy"))
    speaker = engine.prepare_speaker(_profile())

    short_seg = engine.generate_one("One word.", speaker)
    long_seg = engine.generate_one(
        "This line has a great many more words in it than the short one does.",
        speaker,
    )
    assert len(long_seg.samples) > len(short_seg.samples)


# --------------------------------------------------------------------------
# Import sanity (torch-free machine)
# --------------------------------------------------------------------------


def test_registry_and_dummy_import_cleanly():
    import charvoice.engines.dummy  # noqa: F401
    import charvoice.engines.registry  # noqa: F401


def test_gpt_sovits_module_imports_and_instantiates_without_torch():
    """Importing and instantiating must succeed with no torch installed.

    Only calling .load() is allowed to need torch -- see module docstring in
    gpt_sovits.py.
    """
    import charvoice.engines.gpt_sovits as gpt_sovits_module

    engine = gpt_sovits_module.GPTSoVITSEngine()
    assert engine is not None
    assert engine.required_profile_fields() == {"reference_audio", "reference_text"}


# --------------------------------------------------------------------------
# F5-TTS: tuning defaults, fingerprint sensitivity, param passthrough
# --------------------------------------------------------------------------


def test_f5_tts_module_imports_and_instantiates_without_torch():
    """Importing and instantiating must succeed with no torch installed --
    only .load() is allowed to need torch, same contract as gpt_sovits.py."""
    import charvoice.engines.f5_tts as f5_module

    engine = f5_module.F5TTSEngine()
    assert engine is not None
    assert engine.required_profile_fields() == {"reference_audio", "reference_text"}


def test_f5_tts_fingerprint_changes_with_any_tuning_value():
    """The regression guard for the original bug: model_fingerprint() used
    to be a hardcoded constant, so changing nfe_step (or any other tuning
    value) never invalidated the segment cache -- a re-run would silently
    serve back old cached audio instead of regenerating with the new
    setting."""
    import charvoice.engines.f5_tts as f5_module

    engine = f5_module.F5TTSEngine()
    base = EngineConfig(name="f5-tts", extra={"nfe_step": 32})
    changed_nfe = EngineConfig(name="f5-tts", extra={"nfe_step": 64})
    changed_model = EngineConfig(name="f5-tts", extra={"nfe_step": 32, "model": "F5TTS_Base"})

    fp_base = engine.model_fingerprint(base)
    fp_nfe = engine.model_fingerprint(changed_nfe)
    fp_model = engine.model_fingerprint(changed_model)

    assert fp_base != fp_nfe
    assert fp_base != fp_model
    assert fp_nfe != fp_model


def test_f5_tts_fingerprint_stable_for_identical_config():
    import charvoice.engines.f5_tts as f5_module

    engine = f5_module.F5TTSEngine()
    cfg = EngineConfig(name="f5-tts", extra={"nfe_step": 48, "cfg_strength": 2.0})

    assert engine.model_fingerprint(cfg) == engine.model_fingerprint(cfg)
    # a second instance with the same config must agree too -- the
    # fingerprint is a pure function of engine_config, not per-instance state
    assert f5_module.F5TTSEngine().model_fingerprint(cfg) == engine.model_fingerprint(cfg)


def test_f5_tts_tuning_defaults_are_quality_first():
    """Confirms the specific defaults the plan calls for, so a future edit
    that accidentally reverts to F5's speed-first library defaults (e.g.
    nfe_step back to 32, remove_silence back to False) is caught here."""
    import charvoice.engines.f5_tts as f5_module

    tuning = f5_module._tuning_from(EngineConfig(name="f5-tts"))

    assert tuning["nfe_step"] == 48
    assert tuning["remove_silence"] is True
    assert tuning["cfg_strength"] == 2.0
    assert tuning["sway_sampling_coef"] == -1.0
    assert tuning["cross_fade_duration"] == 0.15
    assert tuning["target_rms"] == 0.1


def test_f5_tts_tuning_overridable_via_extra():
    import charvoice.engines.f5_tts as f5_module

    tuning = f5_module._tuning_from(EngineConfig(name="f5-tts", extra={"nfe_step": 64}))

    assert tuning["nfe_step"] == 64
    assert tuning["remove_silence"] is True  # untouched keys keep their default


def test_f5_tts_unknown_extra_keys_do_not_leak_into_tuning():
    import charvoice.engines.f5_tts as f5_module

    tuning = f5_module._tuning_from(EngineConfig(name="f5-tts", extra={"some_other_setting": 1}))

    assert "some_other_setting" not in tuning


def test_f5_tts_generate_one_passes_tuning_and_seed_to_infer(monkeypatch):
    """generate_one() must forward every tuning value to F5TTS.infer(), and
    a fixed, derived seed -- not F5's own seed=None default, which would
    make output non-deterministic and silently break the segment cache's
    premise that identical inputs produce identical audio."""
    import charvoice.engines.f5_tts as f5_module

    captured: dict = {}

    class FakeF5:
        def infer(self, **kwargs):
            captured.update(kwargs)
            return np.zeros(100, dtype=np.float32), 24000, None

    engine = f5_module.F5TTSEngine()
    engine._f5tts = FakeF5()
    engine._tuning = f5_module._tuning_from(EngineConfig(name="f5-tts", extra={"nfe_step": 64}))

    profile = VoiceProfile(
        id="peter1",
        engine="f5-tts",
        reference_audio=Path("/tmp/ref.wav"),
        reference_text="hello there",
        speed=1.2,
    )
    handle = engine.prepare_speaker(profile)
    engine.generate_one("Some line of dialogue.", handle)

    assert captured["nfe_step"] == 64
    assert captured["remove_silence"] is True
    assert captured["speed"] == 1.2
    assert captured["ref_text"] == "hello there"
    assert isinstance(captured["seed"], int)


def test_f5_tts_seed_is_deterministic_per_speaker_and_text():
    import charvoice.engines.f5_tts as f5_module

    a1 = f5_module._seed_for("peter1", "Hello there.")
    a2 = f5_module._seed_for("peter1", "Hello there.")
    b = f5_module._seed_for("peter1", "A different line.")
    c = f5_module._seed_for("miles", "Hello there.")

    assert a1 == a2  # same speaker+text -> same seed, every time
    assert a1 != b  # different text -> different seed
    assert a1 != c  # different speaker -> different seed
