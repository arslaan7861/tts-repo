"""Tests for charvoice.audio. CPU-only, no TTS model, no GPU -- see requirements.md section 26."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest
import soundfile as sf

from charvoice.audio import (
    AudioSegment,
    StreamingWavWriter,
    conform,
    export_mp3,
    limit_peak,
    normalize_peak,
    resample,
    silence,
    to_channels,
)
from charvoice.config import AudioConfig
from charvoice.errors import AudioConversionError, AudioError


def test_silence_sample_count_and_zero() -> None:
    seg = silence(duration_ms=500, sample_rate=16000, channels=1)
    assert len(seg.samples) == 8000
    assert seg.sample_rate == 16000
    assert np.all(seg.samples == 0)
    assert seg.samples.dtype == np.float32


def test_silence_channel_shape() -> None:
    seg = silence(duration_ms=100, sample_rate=8000, channels=2)
    assert seg.samples.shape == (800, 2)
    assert seg.channels == 2
    assert np.all(seg.samples == 0)


def test_normalize_peak_matches_target() -> None:
    rng = np.random.default_rng(0)
    samples = (rng.random(1000).astype(np.float32) - 0.5) * 0.2
    seg = AudioSegment(samples=samples, sample_rate=16000)
    out = normalize_peak(seg, target_dbfs=-3.0)
    peak = np.max(np.abs(out.samples))
    peak_dbfs = 20 * np.log10(peak)
    assert peak_dbfs == pytest.approx(-3.0, abs=0.01)


def test_normalize_peak_silence_untouched() -> None:
    seg = silence(duration_ms=100, sample_rate=8000, channels=1)
    out = normalize_peak(seg, target_dbfs=-1.0)
    assert np.all(out.samples == 0)
    assert not np.any(np.isnan(out.samples))


def test_resample_sample_count() -> None:
    sr, target_sr = 16000, 22050
    n = 1000
    samples = np.sin(np.linspace(0, 10, n)).astype(np.float32)
    seg = AudioSegment(samples=samples, sample_rate=sr)
    out = resample(seg, target_sr)
    expected = round(n * target_sr / sr)
    assert abs(len(out.samples) - expected) <= 1
    assert out.sample_rate == target_sr


def test_resample_numpy_fallback_executes_with_no_optional_deps() -> None:
    """Verify the numpy-only fallback path runs without raising.

    This test environment is expected to have neither soxr nor scipy, so
    resample() should already be exercising _resample_numpy_fallback. We
    assert this directly by blocking those imports regardless.
    """
    import builtins

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name in ("soxr", "scipy", "scipy.signal"):
            raise ImportError(f"blocked for test: {name}")
        return real_import(name, *args, **kwargs)

    samples = np.linspace(-1, 1, 500).astype(np.float32)
    seg = AudioSegment(samples=samples, sample_rate=8000)

    builtins.__import__ = blocking_import
    try:
        out = resample(seg, 12000)
    finally:
        builtins.__import__ = real_import

    assert out.sample_rate == 12000
    expected = round(500 * 12000 / 8000)
    assert abs(len(out.samples) - expected) <= 1
    assert not np.any(np.isnan(out.samples))


def test_to_channels_mono_to_stereo_duplicates() -> None:
    samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    seg = AudioSegment(samples=samples, sample_rate=16000)
    out = to_channels(seg, 2)
    assert out.samples.shape == (3, 2)
    assert np.allclose(out.samples[:, 0], samples)
    assert np.allclose(out.samples[:, 1], samples)


def test_to_channels_stereo_to_mono_averages() -> None:
    left = np.array([0.2, 0.4], dtype=np.float32)
    right = np.array([0.0, 0.0], dtype=np.float32)
    samples = np.stack([left, right], axis=1)
    seg = AudioSegment(samples=samples, sample_rate=16000)
    out = to_channels(seg, 1)
    assert out.channels == 1
    assert np.allclose(out.samples, np.array([0.1, 0.2], dtype=np.float32))


def test_streaming_wav_writer_roundtrip(tmp_path) -> None:
    path = tmp_path / "out.wav"
    segs = [
        silence(duration_ms=50, sample_rate=16000, channels=1),
        AudioSegment(samples=np.ones(200, dtype=np.float32) * 0.1, sample_rate=16000),
        silence(duration_ms=30, sample_rate=16000, channels=1),
    ]
    writer = StreamingWavWriter(path, sample_rate=16000, channels=1, subtype="PCM_16")
    total = 0
    for seg in segs:
        writer.write(seg)
        total += len(seg.samples)
    writer.close()

    assert writer.frames_written == total
    data, sr = sf.read(str(path))
    assert sr == 16000
    assert len(data) == total


def test_streaming_wav_writer_context_manager_closes(tmp_path) -> None:
    path = tmp_path / "ctx.wav"
    with StreamingWavWriter(path, sample_rate=8000, channels=1) as writer:
        writer.write(silence(duration_ms=100, sample_rate=8000, channels=1))
        frames = writer.frames_written

    # File must be finalized and readable after the context exits.
    data, sr = sf.read(str(path))
    assert sr == 8000
    assert len(data) == frames


def test_streaming_wav_writer_rejects_mismatched_rate(tmp_path) -> None:
    path = tmp_path / "bad_rate.wav"
    writer = StreamingWavWriter(path, sample_rate=16000, channels=1)
    try:
        with pytest.raises(AudioError):
            writer.write(silence(duration_ms=10, sample_rate=8000, channels=1))
    finally:
        writer.close()


def test_streaming_wav_writer_rejects_mismatched_channels(tmp_path) -> None:
    path = tmp_path / "bad_channels.wav"
    writer = StreamingWavWriter(path, sample_rate=16000, channels=1)
    try:
        with pytest.raises(AudioError):
            writer.write(silence(duration_ms=10, sample_rate=16000, channels=2))
    finally:
        writer.close()


def test_export_mp3_missing_ffmpeg_raises(tmp_path, monkeypatch) -> None:
    def fake_run(cmd, capture_output=True, text=True):
        raise FileNotFoundError("ffmpeg not found")

    monkeypatch.setattr(subprocess, "run", fake_run)

    wav_path = tmp_path / "in.wav"
    sf.write(str(wav_path), np.zeros(100, dtype=np.float32), 16000)

    with pytest.raises(AudioConversionError):
        export_mp3(wav_path, tmp_path / "out.mp3")


def test_export_mp3_nonzero_exit_raises(tmp_path, monkeypatch) -> None:
    def fake_run(cmd, capture_output=True, text=True):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    wav_path = tmp_path / "in2.wav"
    sf.write(str(wav_path), np.zeros(100, dtype=np.float32), 16000)

    with pytest.raises(AudioConversionError):
        export_mp3(wav_path, tmp_path / "out2.mp3")


def test_conform_round_trips_to_config() -> None:
    cfg = AudioConfig(sample_rate=22050, channels=2, peak_dbfs=-2.0)
    rng = np.random.default_rng(1)
    samples = (rng.random(1000).astype(np.float32) - 0.5) * 0.1
    seg = AudioSegment(samples=samples, sample_rate=16000)

    out = conform(seg, cfg)

    assert out.sample_rate == cfg.sample_rate
    assert out.channels == cfg.channels
    peak = np.max(np.abs(out.samples))
    assert 20 * np.log10(peak) == pytest.approx(cfg.peak_dbfs, abs=0.01)


def test_conform_skips_peak_normalization_when_peak_dbfs_is_none() -> None:
    """peak_dbfs=None is how an engine that already manages its own output
    level (e.g. F5-TTS's internal target_rms) avoids conform() re-scaling
    each segment independently -- which would otherwise make loudness jump
    line to line and amplify noise in quiet segments."""
    cfg = AudioConfig(sample_rate=16000, channels=1, peak_dbfs=None)
    rng = np.random.default_rng(2)
    samples = (rng.random(1000).astype(np.float32) - 0.5) * 0.1
    seg = AudioSegment(samples=samples, sample_rate=16000)

    out = conform(seg, cfg)

    # same sample rate/channels, so no resample/channel-conform touched the
    # values either -- the samples should be untouched, byte for byte
    np.testing.assert_array_equal(out.samples, seg.samples)


def test_conform_still_resamples_when_peak_dbfs_is_none() -> None:
    """peak_dbfs=None only skips the peak step -- resample/channel-conform
    still run."""
    cfg = AudioConfig(sample_rate=8000, channels=1, peak_dbfs=None)
    seg = AudioSegment(samples=np.zeros(16000, dtype=np.float32), sample_rate=16000)

    out = conform(seg, cfg)

    assert out.sample_rate == 8000
    assert len(out.samples) == pytest.approx(8000, abs=1)


def test_limit_peak_leaves_quiet_signal_untouched() -> None:
    """The bug this function fixes: normalize_peak always rescales to hit
    its target exactly, which pulled a signal already under a loudness
    normalization's resulting peak back UP to the ceiling -- overshooting
    the loudness target that was just achieved. limit_peak must never raise
    the level."""
    samples = np.full(100, 0.1, dtype=np.float32)  # peak ~-20 dBFS
    seg = AudioSegment(samples=samples, sample_rate=16000)

    out = limit_peak(seg, ceiling_dbfs=-1.0)

    np.testing.assert_array_equal(out.samples, seg.samples)


def test_limit_peak_scales_down_when_over_ceiling() -> None:
    samples = np.full(100, 0.9, dtype=np.float32)  # peak ~-0.9 dBFS
    seg = AudioSegment(samples=samples, sample_rate=16000)

    out = limit_peak(seg, ceiling_dbfs=-6.0)

    peak = np.max(np.abs(out.samples))
    assert 20 * np.log10(peak) == pytest.approx(-6.0, abs=0.01)


def test_limit_peak_leaves_silence_untouched() -> None:
    seg = AudioSegment(samples=np.zeros(100, dtype=np.float32), sample_rate=16000)

    out = limit_peak(seg, ceiling_dbfs=-1.0)

    assert np.max(np.abs(out.samples)) == 0.0
