"""The API in a few lines.  Run with:  python examples/basic_usage.py"""
import numpy as np

import msadpcm

rate = 22050
t = np.arange(rate * 2) / rate
tone = (0.4 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

# 1. Raw blocks in memory --------------------------------------------------
blocks = msadpcm.encode(tone, block_align=512)           # bytes
pcm = msadpcm.decode(blocks, channels=1, block_align=512, n_samples=len(tone))
print("raw: %d -> %d bytes, SNR %.1f dB" % (tone.nbytes, len(blocks), msadpcm.snr_db(tone, pcm[:, 0])))

# 2. A complete WAV file ----------------------------------------------------
msadpcm.write_wav("tone_adpcm.wav", tone, rate, quality="high")   # block size like Windows ACM
pcm, rate_back = msadpcm.read_wav("tone_adpcm.wav")                # (frames, channels) int16
print("wav: %d frames at %d Hz" % (len(pcm), rate_back))

# 3. Odd block sizes for games / hardware: 128 samples per block -> 70-byte mono blocks
msadpcm.write_wav("tone_128.wav", tone, rate, samples_per_block=128)
print(msadpcm.wav_info("tone_128.wav"))

# 4. Stereo, floats in [-1, 1] are accepted too
stereo = np.stack([tone, tone[::-1]], axis=1).astype(np.float32) / 32768
data = msadpcm.encode(stereo, block_align=1024, quality="fast")
print("stereo blocks:", len(data) // 1024)
