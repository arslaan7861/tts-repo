from __future__ import annotations

from pathlib import Path

import charvoice.colab as colab_module
from charvoice.colab import (
    _GPT_SOVITS_PACKAGE_MARKER,
    _GPT_SOVITS_WEIGHTS_MARKER,
    _seed_drive_project_dir,
    fetch_model,
    in_colab,
    mount_drive,
    offer_download,
)
from charvoice.config import EngineConfig


def test_in_colab_false_locally():
    assert in_colab() is False


def test_mount_drive_returns_none_outside_colab():
    assert mount_drive() is None


def test_offer_download_does_not_raise_outside_colab(tmp_path):
    f = tmp_path / "voiceover.wav"
    f.write_bytes(b"not real audio")
    offer_download(f)  # just prints; must not raise


def test_seed_copies_voices_scripts_config(tmp_path):
    repo_dir = tmp_path / "repo"
    (repo_dir / "voices" / "narrator").mkdir(parents=True)
    (repo_dir / "voices" / "narrator" / "profile.yaml").write_text("engine: dummy\n")
    (repo_dir / "scripts").mkdir()
    (repo_dir / "scripts" / "example.txt").write_text("NARRATOR: hi.\n")
    (repo_dir / "config.yaml").write_text("project:\n  name: x\n")

    project_dir = tmp_path / "drive_project"
    project_dir.mkdir()

    _seed_drive_project_dir(project_dir, repo_dir)

    assert (project_dir / "voices" / "narrator" / "profile.yaml").is_file()
    assert (project_dir / "scripts" / "example.txt").is_file()
    assert (project_dir / "config.yaml").is_file()


def test_seed_is_a_noop_when_voices_already_exists(tmp_path):
    """An existing Drive voices/ (the user's own edits) must never be overwritten."""
    repo_dir = tmp_path / "repo"
    (repo_dir / "voices" / "narrator").mkdir(parents=True)
    (repo_dir / "voices" / "narrator" / "profile.yaml").write_text("engine: dummy\n")

    project_dir = tmp_path / "drive_project"
    custom_dir = project_dir / "voices" / "custom-character"
    custom_dir.mkdir(parents=True)
    (custom_dir / "profile.yaml").write_text("engine: gpt-sovits\n")

    _seed_drive_project_dir(project_dir, repo_dir)

    # the user's own profile is untouched, and narrator was NOT copied in
    assert (project_dir / "voices" / "custom-character" / "profile.yaml").is_file()
    assert not (project_dir / "voices" / "narrator").exists()


def test_seed_skips_missing_sources_without_raising(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()  # empty repo checkout, nothing to seed

    project_dir = tmp_path / "drive_project"
    project_dir.mkdir()

    _seed_drive_project_dir(project_dir, repo_dir)  # must not raise
    assert not (project_dir / "voices").exists()


def test_fetch_model_gpt_sovits_skips_present_sources(tmp_path, monkeypatch):
    """Re-running the fetch cell must not re-clone anything already present."""
    model_dir = tmp_path / "gpt-sovits"
    package_marker = model_dir / _GPT_SOVITS_PACKAGE_MARKER
    weights_marker = model_dir / _GPT_SOVITS_WEIGHTS_MARKER
    package_marker.parent.mkdir(parents=True)
    package_marker.write_text("# stand-in for TTS.py\n")
    weights_marker.parent.mkdir(parents=True)
    weights_marker.write_bytes(b"stand-in checkpoint bytes")

    calls: list[list[str]] = []
    monkeypatch.setattr(colab_module, "_run", lambda cmd, **kw: calls.append(cmd))

    result = fetch_model("gpt-sovits", EngineConfig(name="gpt-sovits", model_dir=str(model_dir)))

    assert result == model_dir
    assert calls == []  # nothing needed fetching, so no subprocess ran


def test_fetch_model_gpt_sovits_clones_missing_sources(tmp_path, monkeypatch):
    model_dir = tmp_path / "gpt-sovits"

    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        # simulate the clone having populated the marker file, like real
        # git clone would, so the function's own follow-up checks pass
        if cmd[:2] == ["git", "clone"] and cmd[-2] == colab_module._GPT_SOVITS_REPO_URL:
            dest = Path(cmd[-1])
            (dest / _GPT_SOVITS_PACKAGE_MARKER).parent.mkdir(parents=True, exist_ok=True)
            (dest / _GPT_SOVITS_PACKAGE_MARKER).write_text("stub")
        elif cmd[:2] == ["git", "clone"] and cmd[-2] == colab_module._GPT_SOVITS_WEIGHTS_URL:
            dest = Path(cmd[-1])
            marker_rel = _GPT_SOVITS_WEIGHTS_MARKER
            (dest / marker_rel).parent.mkdir(parents=True, exist_ok=True)
            (dest / marker_rel).write_bytes(b"stub")

    monkeypatch.setattr(colab_module, "_run", fake_run)

    result = fetch_model("gpt-sovits", EngineConfig(name="gpt-sovits", model_dir=str(model_dir)))

    assert result == model_dir
    assert (model_dir / _GPT_SOVITS_PACKAGE_MARKER).is_file()
    assert (model_dir / _GPT_SOVITS_WEIGHTS_MARKER).is_file()
    assert any(cmd[:2] == ["git", "clone"] for cmd in calls)
