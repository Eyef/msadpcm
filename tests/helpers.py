"""Shared test helpers: signal generators and an independent reference decoder.

``reference_decode`` is deliberately written as plain scalar Python that mirrors
the C decoder in FFmpeg (``adpcm_ms_expand_nibble``) line by line, including C
integer division that truncates toward zero.  It shares no code with the
vectorised decoder in :mod:`msadpcm.codec`.
"""
import struct

import numpy as np

ADAPT = (230, 230, 230, 230, 307, 409, 512, 614, 768, 614, 512, 409, 307, 230, 230, 230)
COEFFS = ((256, 0), (512, -256), (0, 0), (192, 64), (240, 0), (460, -208), (392, -232))


def c_div(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def reference_decode(data: bytes, channels: int, block_align: int, coeffs=COEFFS) -> np.ndarray:
    frames = []
    for off in range(0, len(data) - block_align + 1, block_align):
        blk = data[off:off + block_align]
        if channels == 1:
            pred_idx = [blk[0]]
            delta = [struct.unpack_from("<h", blk, 1)[0]]
            s1 = [struct.unpack_from("<h", blk, 3)[0]]
            s2 = [struct.unpack_from("<h", blk, 5)[0]]
            body = blk[7:]
        else:
            pred_idx = [blk[0], blk[1]]
            delta = list(struct.unpack_from("<hh", blk, 2))
            s1 = list(struct.unpack_from("<hh", blk, 6))
            s2 = list(struct.unpack_from("<hh", blk, 10))
            body = blk[14:]
        frames.append([s2[c] for c in range(channels)])
        frames.append([s1[c] for c in range(channels)])

        def step(c, nib):
            c1, c2 = coeffs[pred_idx[c]]
            p = c_div(s1[c] * c1 + s2[c] * c2, 256)
            p += (nib - 16 if nib & 8 else nib) * delta[c]
            p = max(-32768, min(32767, p))
            s2[c], s1[c] = s1[c], p
            delta[c] = max(16, (ADAPT[nib] * delta[c]) >> 8)
            return p

        for byte in body:
            if channels == 1:
                frames.append([step(0, byte >> 4)])
                frames.append([step(0, byte & 15)])
            else:
                frames.append([step(0, byte >> 4), step(1, byte & 15)])
    return np.array(frames, dtype=np.int16).reshape(-1, channels)


# --------------------------------------------------------------------------
# Synthetic signals (no audio assets needed)
# --------------------------------------------------------------------------
def _i16(x):
    return np.clip(x, -32768, 32767).astype(np.int16)


def voice(n, sr=22050, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    f0 = 130 + 50 * np.sin(2 * np.pi * 0.6 * t)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    s = sum(np.sin(k * ph + 0.3 * k) / k ** 1.1 for k in range(1, 25))
    env = np.clip(np.sin(2 * np.pi * 1.1 * t), 0, 1) ** 0.7
    return _i16((s * env * 0.3 + 0.003 * rng.standard_normal(n)) * 32767)


def music(n, sr=22050, seed=1):
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    notes = rng.choice([220, 262, 330, 392, 440, 523, 659], size=int(n / sr * 3) + 1)
    for i, f in enumerate(notes):
        a = int(i * sr / 3)
        if a >= n:
            break
        m = min(n - a, int(sr * 1.2))
        tt = np.arange(m) / sr
        s[a:a + m] += np.exp(-tt * 3.5) * (np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(4 * np.pi * f * tt))
    return _i16(s / max(1.0, np.abs(s).max()) * 0.75 * 32767)


def noise(n, sr=22050, seed=2):
    return _i16(np.random.default_rng(seed).standard_normal(n) * 6000)


def full_scale_noise(n, seed=3):
    """Worst case for delta growth and clipping."""
    return np.random.default_rng(seed).integers(-32768, 32768, n).astype(np.int16)


def square(n, period=64):
    return np.where((np.arange(n) // (period // 2)) % 2 == 0, 32767, -32768).astype(np.int16)


def impulses(n, every=500):
    x = np.zeros(n, dtype=np.int16)
    x[::every] = 32767
    x[every // 2::every] = -32768
    return x


SIGNALS = {"voice": voice, "music": music, "noise": noise}


def make_stereo(n, sr=22050):
    return np.stack([voice(n, sr), music(n, sr)], axis=1)
