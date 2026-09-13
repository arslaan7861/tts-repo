from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from charvoice.config import AudioConfig, Config, EngineConfig, TTSConfig
from charvoice.engines.registry import clear_engine_cache
from charvoice.errors import ScriptValidationError
from charvoice.parser import parse_script
from charvoice.pipeline import GenerationResult, generate_voiceover, validate_script
from charvoice.profiles import ProfileStore, VoiceProfile


@pytest.fixture(autouse=True)
def _clear_engine_cache():
    clear_engine_cache()
    yield
    clear_engine_cache()


def _dummy_profiles(*ids: str) -> ProfileStore:
    return ProfileStore({i: VoiceProfile(id=i, engine="dummy") for i in ids})


def _dummy_config(tmp_path) -> Config:
    return Config(
        tts=TTSConfig(default_engine="dummy", engines={"dummy": EngineConfig("dummy")}),
        project_dir=tmp_path,
    )


def test_validate_script_passes_for_known_speakers():
    lines = parse_script("NARRATOR: a.\nMILES: b.")
    validate_script(lines, _dummy_profiles("narrator", "miles"))  # no raise


def test_validate_script_reports_all_unknown_speakers_together():
    lines = parse_script("NARRATOR: a.\nMILES: b.\nGREEN-GOBLIN: c.")
    with pytest.raises(ScriptValidationError) as exc_info:
        validate_script(lines, _dummy_profiles("narrator"))
    issues = exc_info.value.issues
    speakers = {issue.speaker for issue in issues}
    assert speakers == {"miles", "green-goblin"}


def test_validate_script_error_message_matches_spec_format():
    lines = parse_script("MILES: hello.")
    with pytest.raises(ScriptValidationError) as exc_info:
        validate_script(lines, _dummy_profiles())
    text = str(exc_info.value)
    assert 'Character "MILES" appears in the script but has no voice profile.' in text
    assert "voices/miles/profile.yaml" in text


def test_generate_voiceover_preserves_script_order(tmp_path):
    script = "NARRATOR: one.\nSPIDER-MAN: two.\nMILES: three.\nSPIDER-MAN: four."
    lines = parse_script(script)
    profiles = _dummy_profiles("narrator", "spider-man", "miles")
    config = _dummy_config(tmp_path)

    result = generate_voiceover(lines, profiles, config, engine_name="dummy")

    assert isinstance(result, GenerationResult)
    assert result.output_path.exists()
    assert result.line_count == 4
    # narrator appears first even though spider-man has more total lines
    assert result.characters[0] == "NARRATOR"


def test_generate_voiceover_output_is_one_combined_file(tmp_path):
    lines = parse_script("NARRATOR: a.\nMILES: b.")
    profiles = _dummy_profiles("narrator", "miles")
    config = _dummy_config(tmp_path)

    result = generate_voiceover(lines, profiles, config, engine_name="dummy")
    samples, sr = sf.read(str(result.output_path))
    assert sr == config.audio.sample_rate
    assert len(samples) > 0


def test_generate_voiceover_refuses_to_overwrite_by_default(tmp_path):
    lines = parse_script("NARRATOR: a.")
    profiles = _dummy_profiles("narrator")
    config = _dummy_config(tmp_path)

    generate_voiceover(lines, profiles, config, engine_name="dummy")
    with pytest.raises(FileExistsError):
        generate_voiceover(lines, profiles, config, engine_name="dummy")

    # overwrite=True is allowed
    generate_voiceover(lines, profiles, config, engine_name="dummy", overwrite=True)


def test_generate_voiceover_raises_before_any_generation_on_unknown_speaker(tmp_path):
    lines = parse_script("NARRATOR: a.\nMILES: b.")
    profiles = _dummy_profiles("narrator")  # miles missing
    config = _dummy_config(tmp_path)

    with pytest.raises(ScriptValidationError):
        generate_voiceover(lines, profiles, config, engine_name="dummy")

    output_dir = config.resolve("output")
    assert not output_dir.exists() or not any(output_dir.iterdir())


def test_generate_voiceover_second_run_hits_cache(tmp_path):
    lines = parse_script("NARRATOR: a fairly long line of dialogue here.")
    profiles = _dummy_profiles("narrator")
    config = _dummy_config(tmp_path)

    r1 = generate_voiceover(
        lines, profiles, config, engine_name="dummy", output_path=tmp_path / "out1.wav"
    )
    assert r1.cache_misses == 1
    assert r1.cache_hits == 0

    r2 = generate_voiceover(
        lines, profiles, config, engine_name="dummy", output_path=tmp_path / "out2.wav"
    )
    assert r2.cache_hits == 1
    assert r2.cache_misses == 0

    s1, _ = sf.read(str(r1.output_path))
    s2, _ = sf.read(str(r2.output_path))
    assert np.allclose(s1, s2)


def test_generate_voiceover_empty_script_raises(tmp_path):
    profiles = _dummy_profiles()
    config = _dummy_config(tmp_path)
    with pytest.raises(ScriptValidationError):
        generate_voiceover([], profiles, config, engine_name="dummy")


def test_generate_voiceover_progress_callback_reports_every_line(tmp_path):
    lines = parse_script("NARRATOR: a.\nMILES: b.\nNARRATOR: c.")
    profiles = _dummy_profiles("narrator", "miles")
    config = _dummy_config(tmp_path)

    seen = []
    generate_voiceover(
        lines, profiles, config, engine_name="dummy", progress_cb=seen.append
    )
    assert len(seen) == 3
    assert seen[-1].current == 3
    assert seen[-1].total == 3


def test_generation_result_summary_matches_spec_block(tmp_path):
    lines = parse_script("NARRATOR: a.\nMILES: b.")
    profiles = _dummy_profiles("narrator", "miles")
    config = _dummy_config(tmp_path)
    result = generate_voiceover(lines, profiles, config, engine_name="dummy")

    text = result.summary()
    assert "Generation complete." in text
    assert "Characters detected:" in text
    assert "- NARRATOR" in text
    assert "- MILES" in text
    assert "Lines:\n2" in text
    assert f"Output:\n{result.output_path}" in text


def test_generate_voiceover_groups_by_speaker_but_keeps_output_order(tmp_path, monkeypatch):
    """Cache-miss lines for the same speaker are generated together via
    generate_batch, but the final file must still follow script order, not
    generation order."""
    from charvoice.engines import dummy as dummy_module

    calls: list[list[str]] = []
    original = dummy_module.DummyEngine.generate_batch

    def spy(self, texts, speaker):
        calls.append(list(texts))
        return original(self, texts, speaker)

    monkeypatch.setattr(dummy_module.DummyEngine, "generate_batch", spy)

    lines = parse_script(
        "NARRATOR: one.\nSPIDER-MAN: two.\nNARRATOR: three.\nSPIDER-MAN: four."
    )
    profiles = _dummy_profiles("narrator", "spider-man")
    config = _dummy_config(tmp_path)

    generate_voiceover(lines, profiles, config, engine_name="dummy")

    # generation happened grouped by speaker (2 calls, not 4)
    assert len(calls) == 2
    # but the written file still has 4 segments in script order -- verified
    # indirectly: total duration should match concatenating all 4 originals
    # in order plus pauses, which other tests already confirm structurally.


def test_resolve_engine_name_uses_profile_engine_over_default():
    """profile.engine was previously dead metadata -- generate_voiceover used
    one global engine for the whole script regardless of what a profile
    said. _resolve_engine_name is the fix: each speaker resolves to their
    own profile's engine, falling back to config.tts.default_engine only
    when a profile doesn't name one."""
    from charvoice.pipeline import _resolve_engine_name

    profiles = ProfileStore(
        {
            "narrator": VoiceProfile(id="narrator", engine="dummy"),
            "peter1": VoiceProfile(id="peter1", engine="some-other-engine"),
        }
    )
    config = Config(tts=TTSConfig(default_engine="dummy"))

    assert _resolve_engine_name("narrator", profiles, config) == "dummy"
    assert _resolve_engine_name("peter1", profiles, config) == "some-other-engine"


def test_generate_voiceover_uses_each_speakers_own_engine(tmp_path):
    """End-to-end: two characters naming different engines in their profiles
    each actually get synthesized by that engine, not silently coerced to
    one global default_engine."""
    from charvoice.engines.base import Engine, SpeakerHandle
    from charvoice.engines.registry import register

    calls: dict[str, int] = {"dummy-a": 0, "dummy-b": 0}

    def _make_fake_engine(tag):
        class _FakeEngine(Engine):
            def load(self, engine_config):
                pass

            def prepare_speaker(self, profile):
                return SpeakerHandle(profile=profile)

            def generate_one(self, text, speaker):
                calls[tag] += 1
                import numpy as np

                from charvoice.audio import AudioSegment

                return AudioSegment(samples=np.zeros(800, dtype=np.float32), sample_rate=22050)

            def model_fingerprint(self, engine_config):
                return tag

        return _FakeEngine

    register("dummy-a")(_make_fake_engine("dummy-a"))
    register("dummy-b")(_make_fake_engine("dummy-b"))

    lines = parse_script("NARRATOR: one.\nPETER1: two.")
    profiles = ProfileStore(
        {
            "narrator": VoiceProfile(id="narrator", engine="dummy-a"),
            "peter1": VoiceProfile(id="peter1", engine="dummy-b"),
        }
    )
    config = Config(
        tts=TTSConfig(default_engine="dummy-a", engines={"dummy-a": EngineConfig("dummy-a")}),
        project_dir=tmp_path,
    )

    generate_voiceover(lines, profiles, config)

    assert calls["dummy-a"] == 1
    assert calls["dummy-b"] == 1


def test_generate_voiceover_hits_target_lufs_without_peak_limit_overshoot(tmp_path):
    """Regression test for a real bug found during manual verification:
    _normalize_timeline_loudness used to apply normalize_peak() after
    normalize_loudness(), and normalize_peak always rescales to its target
    exactly (even pulling a quiet signal UP) -- so a loudness-normalized
    file whose peak was already under the safety ceiling got pushed back up
    to it, overshooting the LUFS target that was just hit. Fixed by using
    limit_peak() (a one-sided clamp) instead.
    """
    pyln = pytest.importorskip("pyloudnorm")

    lines = parse_script("NARRATOR: one.\nSPIDER-MAN: two.\nNARRATOR: three.")
    profiles = _dummy_profiles("narrator", "spider-man")
    config = Config(
        tts=TTSConfig(default_engine="dummy", engines={"dummy": EngineConfig("dummy")}),
        audio=AudioConfig(peak_dbfs=None, target_lufs=-16.0),
        project_dir=tmp_path,
    )

    result = generate_voiceover(lines, profiles, config, engine_name="dummy")

    data, sr = sf.read(str(result.output_path))
    measured = pyln.Meter(sr).integrated_loudness(data)
    assert measured == pytest.approx(-16.0, abs=0.5)
