"""Interoperability with FFmpeg (skipped when ffmpeg is not installed)."""
import shutil
import subprocess

import numpy as np
import pytest

import msadpcm

import helpers

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def ffmpeg_decode(path, channels):
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-acodec", "pcm_s16le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype="<i2").reshape(-1, channels)


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("kw", [dict(samples_per_block=128), dict(block_align=512), dict(block_align=2048)])
def test_ffmpeg_decodes_our_files_bit_exactly(tmp_path, channels, kw):
    pcm = helpers.make_stereo(9000)[:, :channels]
    p = tmp_path / "ours.wav"
    msadpcm.write_wav(p, pcm, 22050, quality="normal", **kw)
    ours, _ = msadpcm.read_wav(p)
    theirs = ffmpeg_decode(p, channels)[: len(ours)]
    assert np.array_equal(ours, theirs)


@pytest.mark.parametrize("channels", [1, 2])
def test_we_decode_ffmpeg_files_bit_exactly(tmp_path, channels):
    pcm = helpers.make_stereo(9000)[:, :channels]
    src, dst = tmp_path / "src.wav", tmp_path / "ff.wav"
    msadpcm.write_wav(src, pcm, 22050, adpcm=False)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-acodec", "adpcm_ms", str(dst)],
                   check=True)
    ours, _ = msadpcm.read_wav(dst)
    theirs = ffmpeg_decode(dst, channels)
    n = min(len(ours), len(theirs))
    assert n >= 9000 and np.array_equal(ours[:n], theirs[:n])


def test_our_encoder_beats_ffmpeg_encoder(tmp_path):
    pcm = helpers.music(22050 * 3)
    src, dst = tmp_path / "src.wav", tmp_path / "ff.wav"
    msadpcm.write_wav(src, pcm, 22050, adpcm=False)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-acodec", "adpcm_ms",
                    "-block_size", "256", str(dst)], check=True)
    ff = msadpcm.read_wav(dst)[0][: len(pcm), 0]
    ours = msadpcm.decode(msadpcm.encode(pcm, block_align=256), channels=1, block_align=256)[: len(pcm), 0]
    assert msadpcm.snr_db(pcm, ours) > msadpcm.snr_db(pcm, ff)
