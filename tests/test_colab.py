from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import charvoice.colab as colab_module
from charvoice.colab import (
    _GPT_SOVITS_PACKAGE_MARKER,
    _GPT_SOVITS_WEIGHTS_MARKER,
    _run,
    _seed_drive_project_dir,
    fetch_model,
    in_colab,
    mount_drive,
    offer_download,
)
from charvoice.config import EngineConfig


def test_in_colab_false_locally():
    assert in_colab() is False


def test_run_succeeds_and_prints_output(capsys):
    result = _run(["python3", "-c", "print('hello from subprocess')"])
    assert result.returncode == 0
    assert "hello from subprocess" in capsys.readouterr().out


def test_run_failure_carries_stderr_in_the_raised_error(capsys):
    """The actual bug: plain subprocess.run(check=True) raised a
    CalledProcessError with no captured output, so a failing pip/git command
    showed only an exit code in the traceback -- never the real reason."""
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        _run(["python3", "-c", "import sys; sys.stderr.write('boom'); sys.exit(1)"])

    assert exc_info.value.returncode == 1
    assert "boom" in exc_info.value.stderr
    # also printed live to stdout, not just attached to the exception
    assert "boom" in capsys.readouterr().out


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


def test_seed_does_not_overwrite_an_existing_character_folder(tmp_path):
    """A Drive character folder -- the user's own edits -- is never touched,
    even though missing characters are still added alongside it (per-folder
    seeding, not whole-or-nothing)."""
    repo_dir = tmp_path / "repo"
    (repo_dir / "voices" / "narrator").mkdir(parents=True)
    (repo_dir / "voices" / "narrator" / "profile.yaml").write_text("engine: dummy\n")

    project_dir = tmp_path / "drive_project"
    custom_dir = project_dir / "voices" / "narrator"
    custom_dir.mkdir(parents=True)
    (custom_dir / "profile.yaml").write_text("engine: gpt-sovits  # hand-edited\n")

    _seed_drive_project_dir(project_dir, repo_dir)

    # the user's hand-edited narrator profile is untouched
    text = (project_dir / "voices" / "narrator" / "profile.yaml").read_text()
    assert "hand-edited" in text


def test_seed_adds_a_character_added_to_the_repo_later(tmp_path):
    """The real bug this guards against: voices/peter1/ added to the repo
    after Drive's voices/ already existed must still reach Drive on the
    next mount, not be silently skipped because voices/ already exists."""
    repo_dir = tmp_path / "repo"
    (repo_dir / "voices" / "narrator").mkdir(parents=True)
    (repo_dir / "voices" / "narrator" / "profile.yaml").write_text("engine: dummy\n")
    (repo_dir / "voices" / "peter1").mkdir(parents=True)
    (repo_dir / "voices" / "peter1" / "profile.yaml").write_text("engine: gpt-sovits\n")

    project_dir = tmp_path / "drive_project"
    existing = project_dir / "voices" / "narrator"
    existing.mkdir(parents=True)
    (existing / "profile.yaml").write_text("engine: dummy  # already on drive\n")

    _seed_drive_project_dir(project_dir, repo_dir)

    assert "already on drive" in (project_dir / "voices" / "narrator" / "profile.yaml").read_text()
    assert (project_dir / "voices" / "peter1" / "profile.yaml").is_file()


def test_seed_skips_missing_sources_without_raising(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()  # empty repo checkout, nothing to seed

    project_dir = tmp_path / "drive_project"
    project_dir.mkdir()

    _seed_drive_project_dir(project_dir, repo_dir)  # must not raise


def test_seed_refreshes_a_character_folder_untouched_since_last_seed(tmp_path):
    """The bug this guards against: a repo update (new engine, new
    reference audio) must reach Drive on a later mount, as long as the
    user never touched that Drive copy themselves -- otherwise every repo
    fix silently stops working for anyone whose Drive was seeded before
    the fix landed."""
    repo_dir = tmp_path / "repo"
    (repo_dir / "voices" / "miles").mkdir(parents=True)
    (repo_dir / "voices" / "miles" / "profile.yaml").write_text("engine: gpt-sovits\n")

    project_dir = tmp_path / "drive_project"
    project_dir.mkdir()

    _seed_drive_project_dir(project_dir, repo_dir)
    assert "gpt-sovits" in (project_dir / "voices" / "miles" / "profile.yaml").read_text()

    # repo is updated (e.g. a later git pull) -- Drive's copy is untouched
    (repo_dir / "voices" / "miles" / "profile.yaml").write_text("engine: f5-tts\n")

    _seed_drive_project_dir(project_dir, repo_dir)

    assert "f5-tts" in (project_dir / "voices" / "miles" / "profile.yaml").read_text()


def test_seed_still_protects_a_character_folder_edited_after_seeding(tmp_path):
    """The existing guarantee must survive the refresh logic: once the user
    edits a Drive copy, a later repo update must never overwrite it, even
    though it was seeded (and therefore has a manifest entry) earlier."""
    repo_dir = tmp_path / "repo"
    (repo_dir / "voices" / "miles").mkdir(parents=True)
    (repo_dir / "voices" / "miles" / "profile.yaml").write_text("engine: gpt-sovits\n")

    project_dir = tmp_path / "drive_project"
    project_dir.mkdir()

    _seed_drive_project_dir(project_dir, repo_dir)

    # user hand-edits the Drive copy
    (project_dir / "voices" / "miles" / "profile.yaml").write_text(
        "engine: gpt-sovits  # my own tweak\n"
    )

    # repo is updated
    (repo_dir / "voices" / "miles" / "profile.yaml").write_text("engine: f5-tts\n")

    _seed_drive_project_dir(project_dir, repo_dir)

    assert "my own tweak" in (project_dir / "voices" / "miles" / "profile.yaml").read_text()


def test_seed_refreshes_config_yaml_untouched_since_last_seed(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "config.yaml").write_text("tts:\n  default_engine: gpt-sovits\n")

    project_dir = tmp_path / "drive_project"
    project_dir.mkdir()

    _seed_drive_project_dir(project_dir, repo_dir)
    assert "gpt-sovits" in (project_dir / "config.yaml").read_text()

    (repo_dir / "config.yaml").write_text("tts:\n  default_engine: f5-tts\n")
    _seed_drive_project_dir(project_dir, repo_dir)

    assert "f5-tts" in (project_dir / "config.yaml").read_text()


def test_seed_still_protects_a_hand_edited_config_yaml(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "config.yaml").write_text("tts:\n  default_engine: gpt-sovits\n")

    project_dir = tmp_path / "drive_project"
    project_dir.mkdir()

    _seed_drive_project_dir(project_dir, repo_dir)
    (project_dir / "config.yaml").write_text("tts:\n  default_engine: dummy  # my choice\n")

    (repo_dir / "config.yaml").write_text("tts:\n  default_engine: f5-tts\n")
    _seed_drive_project_dir(project_dir, repo_dir)

    assert "my choice" in (project_dir / "config.yaml").read_text()


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


def test_filtered_requirements_drops_skipped_packages(tmp_path, monkeypatch):
    """opencc (forced to build from source by --no-binary=opencc) and
    python_mecab_ko (needs the system mecab-ko library) both fail to build
    on a stock Colab runtime and are only needed for zh/ko, which charvoice's
    gpt-sovits adapter never uses. See _GPT_SOVITS_SKIP_REQUIREMENTS."""
    # Pinned separately (see _PIN_TO_INSTALLED); stubbed to None here so this
    # test covers only the skip behavior, independent of the environment.
    monkeypatch.setattr(colab_module, "_installed_version", lambda package: None)

    src = tmp_path / "requirements.txt"
    src.write_text(
        "--no-binary=opencc\n"
        "librosa==0.10.2\n"
        "opencc\n"
        "python_mecab_ko; sys_platform != 'win32'\n"
        "ffmpeg-python\n"
    )

    filtered = colab_module._filtered_requirements(src)

    kept = filtered.read_text().splitlines()
    assert kept == ["librosa==0.10.2", "ffmpeg-python"]
    assert filtered != src  # a separate file; the original is untouched
    assert "opencc" in src.read_text()  # original left as-is


def test_filtered_requirements_pins_transformers(tmp_path):
    """requirements.txt's own transformers<5,>=4.51 resolves to a version
    whose HubertModel import fails inside GPT-SoVITS's cnhubert.py -- a
    documented upstream issue (RVC-Boss/GPT-SoVITS#2687) whose own fix is
    pinning exactly 4.49.0."""
    src = tmp_path / "requirements.txt"
    src.write_text("transformers<5,>=4.51\nx_transformers\n")

    filtered = colab_module._filtered_requirements(src)

    kept = filtered.read_text().splitlines()
    assert "transformers==4.49.0" in kept
    assert "transformers<5,>=4.51" not in kept
    # x_transformers is a different package -- must not be mistaken for a
    # transformers line and rewritten
    assert "x_transformers" in kept


def test_filtered_requirements_pins_numpy_numba_scipy_to_installed(tmp_path, monkeypatch):
    """numpy/numba/scipy ship compiled extensions whose ABI compatibility
    isn't captured by declared version ranges -- letting pip re-resolve any
    of them (e.g. to satisfy requirements.txt's own `numpy<2.0`) can
    silently downgrade numpy while an already-installed numba/scipy stays
    linked against the newer ABI it was actually built with, surfacing as
    'numpy.dtype size changed, may indicate binary incompatibility' deep
    inside an unrelated import. Pinning all three to whatever's already
    installed is the only way to guarantee this install never touches them."""

    def fake_installed_version(package):
        return {"numpy": "2.5.3", "numba": "0.61.2", "scipy": "1.16.3"}.get(package)

    monkeypatch.setattr(colab_module, "_installed_version", fake_installed_version)

    src = tmp_path / "requirements.txt"
    src.write_text("numpy<2.0\nscipy\nnumba\nlibrosa==0.10.2\n")

    filtered = colab_module._filtered_requirements(src)

    kept = filtered.read_text().splitlines()
    assert kept == ["numpy==2.5.3", "scipy==1.16.3", "numba==0.61.2", "librosa==0.10.2"]


def test_filtered_requirements_leaves_uninstalled_pin_targets_alone(tmp_path, monkeypatch):
    """If numpy/numba/scipy aren't installed at all (e.g. local dev without
    the colab extra), the original line passes through unpinned rather than
    being rewritten to a nonsense 'package==None'."""
    monkeypatch.setattr(colab_module, "_installed_version", lambda package: None)

    src = tmp_path / "requirements.txt"
    src.write_text("numpy<2.0\nscipy\n")

    filtered = colab_module._filtered_requirements(src)

    assert filtered.read_text().splitlines() == ["numpy<2.0", "scipy"]


def test_fetch_model_gpt_sovits_installs_filtered_requirements(tmp_path, monkeypatch):
    """TTS.py imports ~35 packages (ffmpeg-python, librosa, ...) our colab
    extra doesn't vendor; GPT-SoVITS's own requirements.txt (minus the
    packages that fail to build, see above) is installed whenever present,
    not just right after a fresh clone, so it self-heals a package dir
    cloned before this step existed or left mid-install."""
    model_dir = tmp_path / "gpt-sovits"
    package_marker = model_dir / _GPT_SOVITS_PACKAGE_MARKER
    weights_marker = model_dir / _GPT_SOVITS_WEIGHTS_MARKER
    package_marker.parent.mkdir(parents=True)
    package_marker.write_text("stub")
    weights_marker.parent.mkdir(parents=True)
    weights_marker.write_bytes(b"stub")
    (model_dir / "requirements.txt").write_text("ffmpeg-python\nlibrosa\nopencc\n")

    calls: list[list[str]] = []
    monkeypatch.setattr(colab_module, "_run", lambda cmd, **kw: calls.append(cmd))

    fetch_model("gpt-sovits", EngineConfig(name="gpt-sovits", model_dir=str(model_dir)))

    pip_calls = [c for c in calls if "pip" in c and "install" in c]
    assert len(pip_calls) == 1
    assert "-r" in pip_calls[0]
    filtered_arg = pip_calls[0][pip_calls[0].index("-r") + 1]
    assert "opencc" not in Path(filtered_arg).read_text()


def test_fetch_model_gpt_sovits_clones_missing_sources(tmp_path, monkeypatch):
    """The GPT-SoVITS package is git-cloned; pretrained weights are downloaded
    file-by-file over plain HTTPS (Colab's base image has no git-lfs, so a
    plain `git clone` of the HuggingFace repo would silently pull LFS pointer
    stubs instead of the real weights -- see colab.py's module comment)."""
    model_dir = tmp_path / "gpt-sovits"

    run_calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        run_calls.append(cmd)
        # simulate the clone having populated the marker file, like real
        # git clone would, so the function's own follow-up checks pass
        if cmd[:2] == ["git", "clone"] and cmd[-2] == colab_module._GPT_SOVITS_REPO_URL:
            dest = Path(cmd[-1])
            (dest / _GPT_SOVITS_PACKAGE_MARKER).parent.mkdir(parents=True, exist_ok=True)
            (dest / _GPT_SOVITS_PACKAGE_MARKER).write_text("stub")

    downloaded: list[tuple[str, Path]] = []

    def fake_download(url, dest):
        downloaded.append((url, dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"stub")

    monkeypatch.setattr(colab_module, "_run", fake_run)
    monkeypatch.setattr(colab_module, "_download", fake_download)
    # fetch_model mutates the real sys.path (it must, so a real TTS.py
    # import would work later) -- restore it so this test doesn't leak
    # entries into others.
    monkeypatch.setattr(sys, "path", list(sys.path))

    result = fetch_model("gpt-sovits", EngineConfig(name="gpt-sovits", model_dir=str(model_dir)))

    assert result == model_dir
    assert (model_dir / _GPT_SOVITS_PACKAGE_MARKER).is_file()
    assert (model_dir / _GPT_SOVITS_WEIGHTS_MARKER).is_file()
    assert any(cmd[:2] == ["git", "clone"] for cmd in run_calls)
    # every weights file was fetched by plain HTTPS, not git
    assert len(downloaded) == len(colab_module._GPT_SOVITS_WEIGHTS_FILES)
    assert all(url.startswith(colab_module._GPT_SOVITS_WEIGHTS_BASE_URL) for url, _ in downloaded)
    assert not any(cmd[:2] == ["git", "lfs"] for cmd in run_calls)
    # TTS.py imports unqualified (`from AR.models...`), which upstream's own
    # api_v2.py satisfies by putting both the repo root and its GPT_SoVITS/
    # subfolder on sys.path -- model_dir IS that repo root here.
    assert str(model_dir) in sys.path
    assert str(model_dir / "GPT_SoVITS") in sys.path


def test_fetch_model_gpt_sovits_resumes_partial_weights(tmp_path, monkeypatch):
    """A re-run after a partial/interrupted download only fetches what's missing.

    The marker file (used for the fast "already done" short-circuit) must
    itself be among the missing files here, otherwise fetch_model correctly
    takes the already-populated fast path instead of checking file-by-file.
    """
    model_dir = tmp_path / "gpt-sovits"
    package_marker = model_dir / _GPT_SOVITS_PACKAGE_MARKER
    package_marker.parent.mkdir(parents=True)
    package_marker.write_text("stub")

    # pre-populate every weights file except the marker itself
    missing = _GPT_SOVITS_WEIGHTS_MARKER
    already_have = [f for f in colab_module._GPT_SOVITS_WEIGHTS_FILES if f != missing]
    for rel_path in already_have:
        dest = model_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"stub")

    downloaded: list[str] = []

    def fake_download(url, dest):
        downloaded.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"stub")

    monkeypatch.setattr(colab_module, "_run", lambda cmd, **kw: None)
    monkeypatch.setattr(colab_module, "_download", fake_download)

    fetch_model("gpt-sovits", EngineConfig(name="gpt-sovits", model_dir=str(model_dir)))

    # only the one missing file was fetched, not all 8 again
    assert len(downloaded) == 1
    assert missing in downloaded[0]
