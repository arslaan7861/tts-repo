"""Colab-only helpers: environment bootstrap, Drive mount, model fetch, download.

Every function here degrades gracefully outside Colab -- this module must
import cleanly with no `google.colab` package present, so the test suite can
run on a plain local machine (requirements.md section 2). Only *calling* a
function outside Colab is refused, and only where it would not make sense
(e.g. mounting a Drive that does not exist).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from charvoice.config import Config, EngineConfig

# Copied from the repo clone into the Drive project folder -- see
# _seed_drive_project_dir. Model checkpoints are deliberately excluded: they
# are large and belong in Drive's own models/ once fetched there directly,
# not duplicated from the repo clone. `voices/` is seeded per-character-folder
# (see below) rather than whole-or-nothing, so a character added to the repo
# later still reaches Drive; scripts/config.yaml are copied once, whole, since
# partially merging a user's edited config.yaml would be worse than skipping it.
_SEED_WHOLE_ENTRIES = ("scripts", "config.yaml")


def in_colab() -> bool:
    """True if running inside a Google Colab runtime."""
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run `cmd`, printing its output live and raising with that output
    attached on failure.

    Plain `subprocess.run(cmd, check=True)` sends stdout/stderr to the
    notebook's output stream directly, but the CalledProcessError it raises
    on failure carries neither -- so a failure here showed only
    "exit status 1" with no way to see *why* it failed. Capturing and
    re-printing keeps the live-output behavior while also attaching the
    captured text to the raised error, so Colab's traceback shows the actual
    pip/git error instead of just the exit code.
    """
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )
    return result


def setup_colab_environment(
    repo_url: str,
    repo_dir: Path | str = "/content/tts-repo",
    branch: str = "main",
    extras: str = "colab",
) -> Path:
    """Clone (or pull) the repo and install it editable. Idempotent.

    Re-running this cell must not re-clone or duplicate state: an existing
    checkout is `git pull`ed instead of re-cloned (requirements.md section 2).
    Returns the repo directory so later cells can reference it.
    """
    repo_dir = Path(repo_dir)

    if in_colab():
        try:
            _run(["nvidia-smi"])
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(
                "WARNING: no GPU detected (nvidia-smi failed). "
                "Request a GPU runtime: Runtime > Change runtime type > GPU."
            )

    if (repo_dir / ".git").is_dir():
        _run(["git", "-C", str(repo_dir), "pull", "--ff-only"])
    else:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--branch", branch, repo_url, str(repo_dir)])

    pip_target = f"{repo_dir}[{extras}]" if extras else str(repo_dir)
    _run([sys.executable, "-m", "pip", "install", "-e", pip_target, "-q"])

    if str(repo_dir / "src") not in sys.path:
        sys.path.insert(0, str(repo_dir / "src"))

    return repo_dir


def _seed_voice_profiles(project_dir: Path, repo_dir: Path) -> None:
    """Copy any character folder from the repo that Drive doesn't have yet.

    Per-folder, not whole-or-nothing: a character added to the repo after
    Drive was first seeded (e.g. a new `voices/peter1/`) must still reach
    Drive on a later mount. An existing Drive character folder -- the user's
    own edits, or a profile already seeded -- is never touched, so this is
    safe to run on every mount, not just the first.
    """
    src_voices = repo_dir / "voices"
    if not src_voices.is_dir():
        return

    dest_voices = project_dir / "voices"
    dest_voices.mkdir(parents=True, exist_ok=True)

    for character_dir in sorted(p for p in src_voices.iterdir() if p.is_dir()):
        dest = dest_voices / character_dir.name
        if dest.exists():
            continue
        shutil.copytree(character_dir, dest)
        print(f"  copied voices/{character_dir.name}")


def _seed_drive_project_dir(project_dir: Path, repo_dir: Path) -> None:
    """Copy starter voices/scripts/config.yaml into the Drive project folder.

    A freshly mounted Drive folder has none of this -- only the git clone
    does. Without seeding, `config.resolve("voices")` would silently point
    at an empty Drive directory and every profile lookup would report
    "no voice profile" even though the repo's profiles are right there (the
    bug this function exists to prevent). Voice profiles are seeded per
    character folder on every call (see `_seed_voice_profiles`); `scripts/`
    and `config.yaml` are copied once, whole, only if Drive doesn't have its
    own copy yet.
    """
    print(f"Seeding {project_dir} with any new starter files from the repo...")
    _seed_voice_profiles(project_dir, repo_dir)

    for name in _SEED_WHOLE_ENTRIES:
        src = repo_dir / name
        dest = project_dir / name
        if not src.exists() or dest.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copyfile(src, dest)
        print(f"  copied {name}")


def mount_drive(
    subpath: str = "character_voice_generator",
    repo_dir: Path | str | None = None,
) -> Path | None:
    """Mount Google Drive and return the project folder under MyDrive.

    Returns None outside Colab or if the user skips mounting -- callers treat
    a None return as "stay on the local checkout", never as an error
    (requirements.md section 13). On first use, seeds the Drive folder with
    the repo's voices/scripts/config.yaml so it starts as a working project
    directory rather than an empty one (pass `repo_dir` -- typically the
    clone from setup_colab_environment -- to enable this).
    """
    if not in_colab():
        print("mount_drive(): not running in Colab, skipping.")
        return None

    from google.colab import drive  # noqa: PLC0415

    drive.mount("/content/drive")
    project_dir = Path("/content/drive/MyDrive") / subpath
    project_dir.mkdir(parents=True, exist_ok=True)

    if repo_dir is not None:
        _seed_drive_project_dir(project_dir, Path(repo_dir))

    return project_dir


_GPT_SOVITS_REPO_URL = "https://github.com/RVC-Boss/GPT-SoVITS.git"
_GPT_SOVITS_WEIGHTS_BASE_URL = "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main"
# Marker files checked per source before (re-)fetching it, so a re-run of
# this cell is a no-op once both are present -- cheap existence checks, no
# hashing, matching requirements.md section 2's "no repeated downloads" rule.
_GPT_SOVITS_PACKAGE_MARKER = "GPT_SoVITS/TTS_infer_pack/TTS.py"
_GPT_SOVITS_WEIGHTS_MARKER = "gsv-v2final-pretrained/s2G2333k.pth"

# The exact files engines/gpt_sovits.py's _checkpoint_paths() resolves, listed
# individually and downloaded over plain HTTPS. Colab's base image has no
# git-lfs (the HF repo's checkpoints are LFS objects, so a plain `git clone`
# there would silently pull pointer stubs, not the actual weights), and this
# repo also bundles v3/v4/v2Pro weights we don't need -- fetching only these
# 8 files is both simpler and smaller than cloning the whole thing.
_GPT_SOVITS_WEIGHTS_FILES = (
    "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "gsv-v2final-pretrained/s2G2333k.pth",
    "chinese-roberta-wwm-ext-large/config.json",
    "chinese-roberta-wwm-ext-large/pytorch_model.bin",
    "chinese-roberta-wwm-ext-large/tokenizer.json",
    "chinese-hubert-base/config.json",
    "chinese-hubert-base/preprocessor_config.json",
    "chinese-hubert-base/pytorch_model.bin",
)


def _download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`, writing to a temp name first (atomic on success)."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:  # noqa: S310
        shutil.copyfileobj(response, f)
    tmp.replace(dest)


# Packages/constraints in GPT-SoVITS's requirements.txt that break a stock
# Colab runtime:
#   - opencc: requirements.txt forces `--no-binary=opencc` (a source build on
#     every platform, bypassing opencc's own prebuilt wheels), which needs a
#     C++ toolchain Colab doesn't ship by default. Used for Chinese text
#     simplified/traditional conversion -- charvoice's gpt-sovits adapter
#     only drives the "en" path (see its _SUPPORTED_LANGUAGES), so this is
#     never imported in practice.
#   - python_mecab_ko: needs the system mecab-ko library, absent on Colab.
#     Used for Korean tokenization -- same "en"-only reasoning as above.
#   - numpy<2.0: force-downgrades Colab's preinstalled numpy, which then
#     breaks transformers==4.49.0's own Hubert import (`module 'numpy.dtypes'
#     has no attribute 'StringDType'`, a NumPy-2.0+-only attribute some
#     transformers-internal lazy import checks for). Dropping this line lets
#     pip keep whatever numpy Colab already has, which is what was already
#     working before this install touched anything.
_GPT_SOVITS_SKIP_REQUIREMENTS = ("opencc", "python_mecab_ko")

# requirements.txt's own range (`transformers<5,>=4.51`) resolves to a
# version newer than GPT-SoVITS actually works with: feature_extractor/
# cnhubert.py's `from transformers import HubertModel` fails on whatever
# 4.57.x pip picks by default, a documented upstream incompatibility
# (https://github.com/RVC-Boss/GPT-SoVITS/issues/2687 -- peft trying to
# import something transformers' internals moved/removed at that version).
# That issue's own fix is exactly this pin.
_GPT_SOVITS_PIN_REQUIREMENTS = {"transformers": "transformers==4.49.0"}

# numpy/numba/scipy ship compiled C extensions built against a specific
# numpy ABI. Their declared version ranges (numba's numpy>=1.24, say) don't
# capture ABI compatibility, only API version -- so letting pip re-resolve
# any of them here can silently downgrade numpy to satisfy some other
# package's older declared range while numba/scipy's ALREADY-INSTALLED
# compiled .so stays linked against Colab's newer numpy ABI (pip sees them
# as "already satisfied" and skips reinstalling). That mismatch surfaces
# deep inside an unrelated import as "numpy.dtype size changed, may
# indicate binary incompatibility" -- exactly the numpy<2.0 line we already
# skip, just via a transitive path instead of requirements.txt's direct one.
# Pinning all three to whatever Colab already has, captured before this
# install runs, is the only way to guarantee the install never touches them
# at all, regardless of what any of the other ~40 lines declares.
_PIN_TO_INSTALLED = ("numpy", "numba", "scipy")


def _installed_version(package: str) -> str | None:
    """The installed version of `package`, or None if it isn't installed."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(package)
    except PackageNotFoundError:
        return None


def _filtered_requirements(requirements_file: Path) -> Path:
    """Copy `requirements_file` with `_GPT_SOVITS_SKIP_REQUIREMENTS` lines
    removed and version-sensitive lines rewritten, so `pip install -r` never
    attempts an opencc/python_mecab_ko build, never resolves transformers to
    a version that breaks GPT-SoVITS's own imports, and never lets numpy/
    numba/scipy drift from whatever Colab already has installed (their
    compiled extensions are ABI-sensitive in a way plain version ranges
    don't capture -- see `_PIN_TO_INSTALLED`). Skip-matching catches both a
    plain requirement line (`opencc`) and a pip directive targeting it
    (`--no-binary=opencc`, which GPT-SoVITS's requirements.txt uses to force
    a source build of opencc specifically -- the thing we're avoiding).
    Returns the path to the filtered copy, written alongside the original.
    """
    pins = dict(_GPT_SOVITS_PIN_REQUIREMENTS)
    for package in _PIN_TO_INSTALLED:
        installed = _installed_version(package)
        if installed:
            pins[package] = f"{package}=={installed}"

    filtered_path = requirements_file.with_name("requirements.charvoice-filtered.txt")
    kept: list[str] = []
    for line in requirements_file.read_text().splitlines():
        normalized = line.strip().lower()
        if any(name in normalized for name in _GPT_SOVITS_SKIP_REQUIREMENTS):
            continue
        pin = next((p for name, p in pins.items() if normalized.startswith(name)), None)
        kept.append(pin if pin else line)
    filtered_path.write_text("\n".join(kept) + "\n")
    return filtered_path


def _fetch_gpt_sovits(model_dir: Path) -> None:
    """Clone GPT-SoVITS, install its own Python deps, and pull v2 weights.

    Three independent steps land in/around the same `model_dir`: the
    upstream repo (needed on sys.path for
    `from GPT_SoVITS.TTS_infer_pack.TTS import TTS`), its own
    `requirements.txt` (TTS.py imports ffmpeg-python, librosa, and ~35 other
    packages our `colab` extra deliberately doesn't vendor -- GPT-SoVITS's
    own requirements file is the actual contract for what it needs), and the
    pretrained checkpoints from HuggingFace (too large to vendor into this
    repo, per requirements.md section 14's "don't hard-code model paths" and
    section 18's model-size guidance). Each step is skipped/re-run
    independently, so an interrupted run resumes instead of starting over.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    if (model_dir / _GPT_SOVITS_PACKAGE_MARKER).is_file():
        print(f"fetch_model(): GPT-SoVITS package already present under {model_dir}.")
    else:
        print(f"fetch_model(): cloning GPT-SoVITS into {model_dir} ...")
        _run(["git", "clone", "--depth", "1", _GPT_SOVITS_REPO_URL, str(model_dir)])

    requirements_file = model_dir / "requirements.txt"
    if requirements_file.is_file():
        # Always run, not just right after a fresh clone: pip skips already
        # -satisfied packages quickly, and this self-heals a package dir that
        # was cloned in an earlier run before this step existed (or was
        # interrupted partway through).
        print("fetch_model(): installing GPT-SoVITS's own Python dependencies ...")
        filtered = _filtered_requirements(requirements_file)
        # No -q here deliberately: this install has already failed once on a
        # source build (see _filtered_requirements), and full pip output is
        # what makes the next such failure diagnosable instead of a bare
        # exit code.
        _run([sys.executable, "-m", "pip", "install", "-r", str(filtered)])

    if (model_dir / _GPT_SOVITS_WEIGHTS_MARKER).is_file():
        print(f"fetch_model(): pretrained v2 weights already present under {model_dir}.")
    else:
        print("fetch_model(): downloading pretrained v2 weights ...")
        for rel_path in _GPT_SOVITS_WEIGHTS_FILES:
            dest = model_dir / rel_path
            if dest.is_file():
                continue
            print(f"  {rel_path}")
            _download(f"{_GPT_SOVITS_WEIGHTS_BASE_URL}/{rel_path}", dest)

    # GPT-SoVITS's own TTS.py imports unqualified (`from AR.models...`,
    # `from tools.audio_sr...`), not `from GPT_SoVITS.AR.models...` -- it
    # expects to be run from its own repo root with that root AND its
    # GPT_SoVITS/ subfolder both on sys.path (confirmed from upstream's own
    # api_v2.py: `sys.path.append(now_dir)` then
    # `sys.path.append(f"{now_dir}/GPT_SoVITS")`, where now_dir is cwd).
    # model_dir IS that repo root here, since we cloned straight into it.
    for path in (model_dir, model_dir / "GPT_SoVITS"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def fetch_model(engine: str, engine_config: EngineConfig) -> Path | None:
    """Ensure the engine's model files are present, downloading if needed.

    Re-running this cell is a no-op once the model is present -- checked via
    cheap marker-file existence, not by re-downloading and diffing
    (requirements.md section 2, section 19).
    """
    if not engine_config.model_dir:
        print(f"fetch_model(): engine {engine!r} has no model_dir configured, nothing to fetch.")
        return None

    model_dir = Path(engine_config.model_dir).expanduser().resolve()

    if engine == "gpt-sovits":
        _fetch_gpt_sovits(model_dir)
        return model_dir

    if model_dir.is_dir() and any(model_dir.iterdir()):
        print(f"fetch_model(): {model_dir} already populated, skipping.")
        return model_dir

    print(
        f"fetch_model(): {model_dir} is empty or missing. Populate it with the "
        f"{engine!r} checkpoint files before generating (requirements.md section 14)."
    )
    return model_dir


def offer_download(path: Path | str) -> None:
    """Hand the final file to the browser for download. No-op outside Colab."""
    path = Path(path)
    if not in_colab():
        print(f"offer_download(): not running in Colab. File is at: {path}")
        return

    from google.colab import files  # noqa: PLC0415

    files.download(str(path))


def project_dir_for(config: Config, drive_dir: Path | None) -> Config:
    """Re-base `config` onto the mounted Drive folder, if one was mounted.

    Threading this explicitly through every call site is what makes Drive
    mounting actually take effect -- leaving paths resolved against `cwd()`
    after mounting Drive is a documented trap (requirements.md section 2):
    the profiles and output would silently keep reading/writing the local
    checkout instead of Drive.
    """
    if drive_dir is None:
        return config
    return config.with_project_dir(drive_dir)
