"""Audio primitives: load/save, resample, channel-conform, normalize, stream to disk.

Pure CPU, no torch. Everything here must run on the local machine with only
the core deps (`numpy`, `soundfile`) installed -- see requirements.md section
2. `soxr`, `scipy` and `pyloudnorm` are optional quality upgrades, imported
lazily so their absence never breaks import or basic functionality.

A script's full audio is never held in memory at once: `StreamingWavWriter`
writes each segment to disk as it is produced (requirements.md sections 8, 19).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from charvoice.config import AudioConfig
from charvoice.errors import AudioConversionError, AudioError


@dataclass(frozen=True)
class AudioSegment:
    samples: np.ndarray  # float32, shape (n,) mono or (n, channels) multi-channel
    sample_rate: int

    @property
    def channels(self) -> int:
        return 1 if self.samples.ndim == 1 else self.samples.shape[1]

    @property
    def duration_s(self) -> float:
        return len(self.samples) / float(self.sample_rate)


def load_segment(path: Path) -> AudioSegment:
    """Read an audio file, always as float32 samples."""
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return AudioSegment(samples=samples, sample_rate=sample_rate)


def save_segment(seg: AudioSegment, path: Path, subtype: str = "PCM_16") -> None:
    """Write a segment to a WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), seg.samples, seg.sample_rate, subtype=subtype)


def _resample_numpy_fallback(samples: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Linear-interpolation resampler, zero dependencies.

    Quality is lower than soxr/scipy's polyphase filters, but it always works,
    which is the point: resample() must never hard-fail for lack of an
    optional dep.
    """
    n = samples.shape[0]
    n_out = round(n * target_sr / sr)
    if n_out <= 0:
        shape = (0,) if samples.ndim == 1 else (0, samples.shape[1])
        return np.zeros(shape, dtype=np.float32)
    src_t = np.arange(n, dtype=np.float64)
    dst_t = np.linspace(0, n - 1, num=n_out, dtype=np.float64) if n > 1 else np.zeros(n_out)
    if samples.ndim == 1:
        return np.interp(dst_t, src_t, samples).astype(np.float32)
    out = np.empty((n_out, samples.shape[1]), dtype=np.float32)
    for c in range(samples.shape[1]):
        out[:, c] = np.interp(dst_t, src_t, samples[:, c])
    return out


def resample(seg: AudioSegment, target_sr: int) -> AudioSegment:
    """Resample to `target_sr`.

    Tries soxr (best quality) then scipy.signal.resample_poly, falling back to
    a numpy-only linear interpolation so this always works with zero optional
    deps installed. Each candidate is imported lazily here, not at module
    top-level, so the optional deps stay truly optional.
    """
    if seg.sample_rate == target_sr:
        return seg

    samples = seg.samples

    try:
        import soxr

        out = soxr.resample(samples, seg.sample_rate, target_sr)
        return AudioSegment(samples=out.astype(np.float32), sample_rate=target_sr)
    except ImportError:
        pass

    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(target_sr, seg.sample_rate)
        up, down = target_sr // g, seg.sample_rate // g
        out = resample_poly(samples, up, down, axis=0)
        return AudioSegment(samples=out.astype(np.float32), sample_rate=target_sr)
    except ImportError:
        pass

    out = _resample_numpy_fallback(samples, seg.sample_rate, target_sr)
    return AudioSegment(samples=out, sample_rate=target_sr)


def to_channels(seg: AudioSegment, target_channels: int) -> AudioSegment:
    """Convert between mono and stereo. Mono->stereo duplicates; stereo->mono averages."""
    if seg.channels == target_channels:
        return seg

    if seg.channels == 1 and target_channels == 2:
        out = np.repeat(seg.samples[:, np.newaxis], 2, axis=1)
        return AudioSegment(samples=out.astype(np.float32), sample_rate=seg.sample_rate)

    if seg.channels == 2 and target_channels == 1:
        out = seg.samples.mean(axis=1)
        return AudioSegment(samples=out.astype(np.float32), sample_rate=seg.sample_rate)

    raise AudioError(
        f"unsupported channel conversion: {seg.channels} -> {target_channels} "
        "(only mono and stereo are supported)"
    )


def normalize_peak(seg: AudioSegment, target_dbfs: float = -1.0) -> AudioSegment:
    """Scale so the peak sample hits `target_dbfs`. Leaves silence untouched."""
    samples = seg.samples
    peak = np.max(np.abs(samples)) if samples.size else 0.0
    if peak > 0:
        scale = (10 ** (target_dbfs / 20.0)) / peak
        samples = (samples * scale).astype(np.float32)
    return AudioSegment(samples=samples, sample_rate=seg.sample_rate)


def normalize_loudness(seg: AudioSegment, target_lufs: float) -> AudioSegment:
    """Normalize integrated loudness to `target_lufs` LUFS via pyloudnorm.

    Lazy-imported so the module stays importable without the optional
    `audio` extra; calling this without it installed raises AudioError with
    an actionable message instead of crashing at import time.
    """
    try:
        import pyloudnorm as pyln
    except ImportError as exc:
        raise AudioError(
            "normalize_loudness() needs the optional 'pyloudnorm' dependency. "
            "Install it with: pip install charvoice[audio]"
        ) from exc

    meter = pyln.Meter(seg.sample_rate)
    loudness = meter.integrated_loudness(seg.samples)
    out = pyln.normalize.loudness(seg.samples, loudness, target_lufs)
    return AudioSegment(samples=out.astype(np.float32), sample_rate=seg.sample_rate)


def silence(duration_ms: int, sample_rate: int, channels: int = 1) -> AudioSegment:
    """Actual zero samples of the right shape/dtype, not a sentinel.

    This keeps StreamingWavWriter.write uniform -- it never needs to
    special-case silence.
    """
    n = round(duration_ms * sample_rate / 1000.0)
    shape = (n,) if channels == 1 else (n, channels)
    return AudioSegment(samples=np.zeros(shape, dtype=np.float32), sample_rate=sample_rate)


def conform(seg: AudioSegment, cfg: AudioConfig) -> AudioSegment:
    """Bring a segment to the config's sample_rate/channels/loudness, in order.

    Order: resample -> channel-conform -> peak-normalize -> loudness-normalize
    (only if cfg.target_lufs is set). Peak normalization always runs so every
    segment has a predictable, consistent level before optional loudness
    matching fine-tunes it.

    Note for pipeline.py: cache keys for synthesized segments must never
    include pause settings. Pauses (pause_between_lines_ms etc.) are generated
    separately via silence() and spliced in at assembly time -- they are never
    baked into a cached segment. A cached line's audio is independent of the
    pauses around it, so keying on pause settings would invalidate the cache
    for no reason.
    """
    seg = resample(seg, cfg.sample_rate)
    seg = to_channels(seg, cfg.channels)
    seg = normalize_peak(seg, cfg.peak_dbfs)
    if cfg.target_lufs is not None:
        seg = normalize_loudness(seg, cfg.target_lufs)
    return seg


class StreamingWavWriter:
    """Writes a WAV incrementally so peak memory is one segment, not the whole script."""

    def __init__(
        self, path: Path, sample_rate: int, channels: int, subtype: str = "PCM_16"
    ) -> None:
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.channels = channels
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = sf.SoundFile(
            str(self.path),
            mode="w",
            samplerate=sample_rate,
            channels=channels,
            subtype=subtype,
        )
        self._frames_written = 0

    def write(self, seg: AudioSegment) -> None:
        """Append a segment. Raises AudioError on rate/channel mismatch.

        This is a safety check, not a silent fix -- the caller should have
        called conform() already.
        """
        if seg.sample_rate != self.sample_rate:
            raise AudioError(
                f"StreamingWavWriter sample rate mismatch: writer is "
                f"{self.sample_rate} Hz, segment is {seg.sample_rate} Hz. "
                "Call conform() before writing."
            )
        if seg.channels != self.channels:
            raise AudioError(
                f"StreamingWavWriter channel mismatch: writer is "
                f"{self.channels}-channel, segment is {seg.channels}-channel. "
                "Call conform() before writing."
            )
        self._file.write(seg.samples)
        self._frames_written += len(seg.samples)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> StreamingWavWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def frames_written(self) -> int:
        return self._frames_written


def export_mp3(wav_path: Path, mp3_path: Path) -> None:
    """Convert a WAV to MP3 by shelling out to ffmpeg.

    The WAV path (soundfile) never shells out to anything; ffmpeg is needed
    only for this optional MP3 export step.
    """
    mp3_path = Path(mp3_path)
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(wav_path), str(mp3_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise AudioConversionError(
            "ffmpeg is not installed or not on PATH. MP3 export requires ffmpeg: "
            "install it and ensure the 'ffmpeg' command is available."
        ) from exc

    if result.returncode != 0:
        raise AudioConversionError(
            f"ffmpeg failed converting {wav_path} to {mp3_path} "
            f"(exit code {result.returncode}):\n{result.stderr}"
        )
