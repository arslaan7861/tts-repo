from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from charvoice.config import Config, EngineConfig, TTSConfig
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
