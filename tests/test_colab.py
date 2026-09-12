from __future__ import annotations

from charvoice.colab import _seed_drive_project_dir, in_colab, mount_drive, offer_download


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
