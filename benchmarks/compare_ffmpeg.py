"""Compare msadpcm's encoder with FFmpeg's ``adpcm_ms`` (needs ffmpeg on PATH).

    python benchmarks/compare_ffmpeg.py

Uses the synthetic signals from tests/helpers.py so the numbers can be
reproduced without any audio assets.  Every file is decoded with msadpcm's
decoder, which is bit-exact with FFmpeg's (checked in tests/test_ffmpeg.py),
so the SNR differences come from the encoders alone.
"""
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))

import msadpcm  # noqa: E402
from helpers import SIGNALS  # noqa: E402

RATE, SECONDS = 22050, 6
PRESETS = ("fast", "normal", "high", "max")


def ffmpeg_encode(pcm, block_align, tmp):
    src, dst = os.path.join(tmp, "src.wav"), os.path.join(tmp, "ff.wav")
    msadpcm.write_wav(src, pcm, RATE, adpcm=False)
    t0 = time.perf_counter()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-acodec", "adpcm_ms",
                    "-block_size", str(block_align), dst], check=True)
    seconds = time.perf_counter() - t0
    decoded, _ = msadpcm.read_wav(dst)
    return msadpcm.snr_db(pcm, decoded[: len(pcm), 0]), seconds


def ours(pcm, block_align, quality):
    t0 = time.perf_counter()
    data = msadpcm.encode(pcm, block_align=block_align, quality=quality)
    seconds = time.perf_counter() - t0
    decoded = msadpcm.decode(data, channels=1, block_align=block_align)
    return msadpcm.snr_db(pcm, decoded[: len(pcm), 0]), seconds


def main():
    header = "%-7s %-6s %-18s" % ("signal", "block", "ffmpeg adpcm_ms") + "".join(
        "%-18s" % ("msadpcm " + p) for p in PRESETS)
    print(header)
    print("-" * len(header))
    with tempfile.TemporaryDirectory() as tmp:
        for block_align in (256, 1024):
            for name, make in SIGNALS.items():
                pcm = make(RATE * SECONDS, RATE)
                cells = ["%5.2f dB %5.2fs" % ffmpeg_encode(pcm, block_align, tmp)]
                cells += ["%5.2f dB %5.2fs" % ours(pcm, block_align, q) for q in PRESETS]
                print("%-7s %-6d " % (name, block_align) + "  ".join("%-16s" % c for c in cells))


if __name__ == "__main__":
    main()
