"""Microsoft ADPCM (WAVE_FORMAT_ADPCM, tag 0x0002) encoder and decoder.

Pure Python + NumPy.  All blocks (and, for the encoder, all candidate
parameter sets) are processed in parallel with vectorised arithmetic, so no
multiprocessing is needed.

Block layout (see the format description in the README):

mono   : predictor(1) delta(2) sample1(2) sample2(2) | nibbles (high first)
stereo : predL(1) predR(1) deltaL(2) deltaR(2) s1L(2) s1R(2) s2L(2) s2R(2)
         | bytes with left nibble in the high half, right nibble in the low half

The decoder follows the reference behaviour (the same as FFmpeg's
``adpcm_ms``): the prediction is ``(s1*c1 + s2*c2) / 256`` with *truncation
towards zero*, the delta update is ``(table[nibble] * delta) >> 8`` with a
lower bound of 16.  The encoder simulates exactly this decoder, so what it
measures is what a decoder will output, bit for bit.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "STANDARD_COEFFS",
    "ADAPTATION_TABLE",
    "QUALITY_LEVELS",
    "samples_per_block",
    "block_align_for",
    "default_block_align",
    "encode",
    "decode",
    "snr_db",
]

#: The seven built-in predictor coefficient pairs (coeff1, coeff2).
STANDARD_COEFFS: Tuple[Tuple[int, int], ...] = (
    (256, 0), (512, -256), (0, 0), (192, 64),
    (240, 0), (460, -208), (392, -232),
)

#: Delta adaptation multipliers (x/256), indexed by the *unsigned* nibble.
ADAPTATION_TABLE: Tuple[int, ...] = (
    230, 230, 230, 230, 307, 409, 512, 614,
    768, 614, 512, 409, 307, 230, 230, 230,
)

MIN_DELTA = 16
#: The header stores the initial delta as a signed 16-bit value.
MAX_HEADER_DELTA = 32767
#: Same overflow guard as FFmpeg: INT_MAX / 768.
_DELTA_CEIL = 2147483647 // 768

_ADAPT = np.asarray(ADAPTATION_TABLE, dtype=np.int32)

ProgressFn = Callable[[int, int], None]
Coeffs = Sequence[Tuple[int, int]]


# --------------------------------------------------------------------------
# Block geometry
# --------------------------------------------------------------------------
def _check_channels(channels: int) -> None:
    if channels not in (1, 2):
        raise ValueError("MS ADPCM supports only 1 or 2 channels, got %r" % (channels,))


def _header_size(channels: int) -> int:
    return 7 * channels


def samples_per_block(block_align: int, channels: int) -> int:
    """Number of sample frames stored in one block of ``block_align`` bytes."""
    _check_channels(channels)
    hdr = _header_size(channels)
    if not hdr < block_align <= 0xFFFF:
        raise ValueError(
            "block_align must be in (%d, 65535] for %d channel(s), got %r"
            % (hdr, channels, block_align)
        )
    return (block_align - hdr) * 2 // channels + 2


def block_align_for(spb: int, channels: int) -> int:
    """Inverse of :func:`samples_per_block`: bytes per block for ``spb`` frames.

    Mono blocks need an even number of samples per block.
    """
    _check_channels(channels)
    if spb < 4:
        raise ValueError("samples per block must be >= 4")
    if channels == 1:
        if spb % 2:
            raise ValueError("mono blocks need an even number of samples per block")
        ba = (spb - 2) // 2 + 7
    else:
        ba = spb - 2 + 14
    if ba > 0xFFFF:
        raise ValueError("block would exceed 65535 bytes")
    return ba


def default_block_align(sample_rate: int, channels: int) -> int:
    """The block size Windows' ACM codec picks: 256 bytes per channel per 11 kHz."""
    _check_channels(channels)
    return 256 * channels * max(1, int(sample_rate) // 11025)


def _resolve_geometry(
    channels: int, block_align: Optional[int], spb: Optional[int]
) -> Tuple[int, int]:
    if block_align is not None and spb is not None:
        raise ValueError("pass either block_align or samples_per_block, not both")
    if spb is not None:
        block_align = block_align_for(spb, channels)
    elif block_align is None:
        block_align = 256 * channels
    return int(block_align), samples_per_block(int(block_align), channels)


def _coeff_arrays(coeffs: Optional[Coeffs]) -> Tuple[np.ndarray, np.ndarray]:
    table = STANDARD_COEFFS if coeffs is None else tuple(coeffs)
    if not 1 <= len(table) <= 256:
        raise ValueError("need between 1 and 256 coefficient pairs")
    arr = np.asarray(table, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("coeffs must be a sequence of (coeff1, coeff2) pairs")
    if np.abs(arr).max() > 32767:
        raise ValueError("coefficients must fit in 16 bits")
    return arr[:, 0].astype(np.int32), arr[:, 1].astype(np.int32)


# --------------------------------------------------------------------------
# Input normalisation
# --------------------------------------------------------------------------
def _as_frames(pcm, channels: Optional[int]) -> np.ndarray:
    """Return int16 array of shape (frames, channels)."""
    if isinstance(pcm, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(pcm, dtype="<i2")
    else:
        arr = np.asarray(pcm)
        if not isinstance(pcm, np.ndarray) and arr.dtype.kind == "i":
            # A plain Python list of ints: the dtype numpy picked is incidental,
            # so accept it when every value fits into int16.  (A real int32/int64
            # *array* stays rejected: its scale, e.g. 24-bit audio, is ambiguous.)
            if arr.size and (arr.min() < -32768 or arr.max() > 32767):
                raise ValueError("integer samples must fit into int16")
            arr = arr.astype(np.int16)

    if arr.dtype.kind == "f":
        arr = np.clip(np.rint(arr.astype(np.float64) * 32768.0), -32768, 32767)
    elif arr.dtype.kind in "iu" and arr.dtype != np.int16:
        if arr.dtype.itemsize != 2 or arr.dtype.kind != "i":
            raise TypeError(
                "integer input must be int16 (got %s); convert it first or pass "
                "floats in [-1, 1]" % arr.dtype
            )
    elif arr.dtype.kind not in "iuf":
        raise TypeError("unsupported dtype %s" % arr.dtype)
    arr = arr.astype(np.int16, copy=False)

    if arr.ndim == 1:
        ch = 1 if channels is None else int(channels)
        _check_channels(ch)
        if arr.size % ch:
            raise ValueError("%d samples do not divide into %d channels" % (arr.size, ch))
        arr = arr.reshape(-1, ch)
    elif arr.ndim == 2:
        if channels is not None and arr.shape[1] != channels:
            raise ValueError("array has %d channels, channels=%r" % (arr.shape[1], channels))
        _check_channels(arr.shape[1])
    else:
        raise ValueError("expected a 1-D or 2-D array")
    return arr


# --------------------------------------------------------------------------
# Quality presets
# --------------------------------------------------------------------------
#: ``n_pred``   - how many predictors are tried per block.  Ranking them by
#:               open-loop error is unreliable (tonal material loses up to
#:               5 dB), so every preset tries all of them.
#: ``mults``    - initial deltas tried, as multiples of the per-block estimate
#: ``refine``   - second pass: fine search around the winning delta
def _grid(*parts) -> Tuple[float, ...]:
    return tuple(sorted({round(float(v), 4) for p in parts for v in p}))


# Every preset's coarse search covers all candidates of the previous preset.
# The refinement pass is centred on each preset's own winner, so the final
# SNR is almost always, but not strictly always, ordered fast <= normal <= high.
_FAST = (0.3, 0.7, 1.4)
_NORMAL = _grid(_FAST, (0.2, 0.4, 1.0, 1.6))
_HIGH = _grid(_NORMAL, (0.15, 0.25, 0.35, 0.5, 2.0, 3.0))
_MAX = _grid(_HIGH, np.geomspace(0.05, 6.0, 28))

QUALITY_LEVELS = {
    "fast": dict(n_pred=7, mults=_FAST, refine=False),
    "normal": dict(n_pred=7, mults=_NORMAL, refine=True),
    "high": dict(n_pred=7, mults=_HIGH, refine=True),
    "max": dict(n_pred=7, mults=_MAX, refine=True),
}

_REFINE_MULTS = (0.8, 0.9, 1.1, 1.25)
#: Rough number of candidate lanes processed per vectorised chunk.
_LANES_PER_CHUNK = 1 << 17
#: Upper bound for ``rows * samples_per_block`` per chunk; keeps memory flat
#: (roughly 120 MB) however large the blocks are.
_ELEMS_PER_CHUNK = 1 << 22


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------
def _trunc_div256(t: np.ndarray) -> np.ndarray:
    """int32 ``t / 256`` truncating toward zero (C semantics)."""
    return (t + ((t >> 31) & 255)) >> 8


def _simulate(
    x: np.ndarray,
    cand_pred: np.ndarray,
    cand_delta: np.ndarray,
    c1_tab: np.ndarray,
    c2_tab: np.ndarray,
    want_nibs: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Encode every row of ``x`` with every candidate.

    x          : (r, N) int32 samples
    cand_pred  : (r, M) predictor index per candidate
    cand_delta : (r, M) initial delta per candidate
    want_nibs  : also return the nibbles; only worth it for a few candidates,
                 the buffer is (N-2) * r * M bytes
    returns    : err (r, M) float64 squared error, nibbles (N-2, r*M) uint8 or None
    """
    r, n = x.shape
    m = cand_pred.shape[1]
    lanes = r * m
    row = np.repeat(np.arange(r), m)
    xt = np.ascontiguousarray(x.T)  # (N, r)

    flat_pred = cand_pred.reshape(-1)
    c1 = c1_tab[flat_pred]
    c2 = c2_tab[flat_pred]
    s1 = xt[1][row].astype(np.int32)
    s2 = xt[0][row].astype(np.int32)
    delta = cand_delta.reshape(-1).astype(np.int32)
    err = np.zeros(lanes, dtype=np.float64)
    over = np.zeros(lanes, dtype=bool)
    nibs = np.empty((n - 2, lanes), dtype=np.uint8) if want_nibs else None

    for i in range(2, n):
        pred = _trunc_div256(s1 * c1 + s2 * c2)
        target = xt[i][row]
        q = np.rint((target - pred) / delta)
        np.clip(q, -8, 7, out=q)
        q = q.astype(np.int32)
        recon = pred + q * delta
        np.clip(recon, -32768, 32767, out=recon)
        e = (target - recon).astype(np.float64)
        err += e * e
        u = q & 15
        if want_nibs:
            nibs[i - 2] = u
        s2 = s1
        s1 = recon
        delta = (_ADAPT[u] * delta) >> 8
        np.maximum(delta, MIN_DELTA, out=delta)
        np.minimum(delta, _DELTA_CEIL, out=delta)
        over |= delta > MAX_HEADER_DELTA

    # Decoders that keep the delta in 16 bits would diverge once it exceeds
    # 32767; steer away from such candidates whenever an alternative exists.
    err += over * 1e18
    return err.reshape(r, m), nibs


def _search_rows(
    x: np.ndarray,
    c1_tab: np.ndarray,
    c2_tab: np.ndarray,
    n_pred: int,
    mults: Sequence[float],
    refine: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find predictor, initial delta and nibbles for each row of ``x``."""
    r, n = x.shape
    n_coef = len(c1_tab)
    n_pred = min(n_pred, n_coef)

    # Open-loop residuals of every predictor (uses the true signal as history).
    h1 = x[:, 1:-1]
    h2 = x[:, :-2]
    tgt = x[:, 2:]
    res = np.empty((n_coef, r, n - 2), dtype=np.int32)
    for p in range(n_coef):
        res[p] = tgt - _trunc_div256(h1 * c1_tab[p] + h2 * c2_tab[p])
    score = np.empty((n_coef, r), dtype=np.float64)
    for p in range(n_coef):  # one predictor at a time: no (P, r, n) float64 temporary
        rp = res[p].astype(np.float64)
        score[p] = np.einsum("ij,ij->i", rp, rp)
    order = np.argsort(score, axis=0)[:n_pred]  # (n_pred, r) best first
    sel = order.T  # (r, n_pred)

    # Initial-delta estimate: scale of the first few residuals of that predictor.
    head = min(16, n - 2)
    rows_idx = np.arange(r)[:, None]
    head_res = np.abs(res[:, :, :head]).astype(np.float64).mean(axis=2)  # (P, r)
    est = head_res.T[rows_idx, sel]  # (r, n_pred)
    est = np.maximum(est, MIN_DELTA)

    mult = np.asarray(mults, dtype=np.float64)
    cand_delta = np.rint(est[:, :, None] * mult[None, None, :])
    cand_delta = np.clip(cand_delta, MIN_DELTA, MAX_HEADER_DELTA).astype(np.int32)
    cand_pred = np.broadcast_to(sel[:, :, None], cand_delta.shape)
    cand_delta = cand_delta.reshape(r, -1)
    cand_pred = np.ascontiguousarray(cand_pred.reshape(r, -1))

    err, _ = _simulate(x, cand_pred, cand_delta, c1_tab, c2_tab)
    ar = np.arange(r)
    best = err.argmin(axis=1)
    best_pred = cand_pred[ar, best]
    best_delta = cand_delta[ar, best]
    best_err = err[ar, best]

    if refine:
        rm = np.asarray(_REFINE_MULTS, dtype=np.float64)
        rdelta = np.rint(best_delta[:, None] * rm[None, :])
        rdelta = np.clip(rdelta, MIN_DELTA, MAX_HEADER_DELTA).astype(np.int32)
        rpred = np.broadcast_to(best_pred[:, None], rdelta.shape).copy()
        err2, _ = _simulate(x, rpred, rdelta, c1_tab, c2_tab)
        b2 = err2.argmin(axis=1)
        e2 = err2[ar, b2]
        better = e2 < best_err
        best_delta = np.where(better, rdelta[ar, b2], best_delta)
        best_err = np.where(better, e2, best_err)

    # Only the winners need their nibbles: one extra pass over r lanes.
    final_err, best_nibs = _simulate(
        x, best_pred[:, None], best_delta[:, None].astype(np.int32), c1_tab, c2_tab, want_nibs=True)
    best_err = final_err[:, 0]
    best_nibs = best_nibs.T  # (r, N-2)

    return (
        best_pred.astype(np.uint8),
        best_delta.astype(np.int32),
        np.ascontiguousarray(best_nibs, dtype=np.uint8),
        best_err,
    )


def _encode_rows(
    rows: np.ndarray,
    coeffs: Optional[Coeffs],
    quality: str,
    progress: Optional[ProgressFn],
):
    """Encode (R, N) int16 rows. Returns preds, deltas, nibbles, errors."""
    if quality not in QUALITY_LEVELS:
        raise ValueError("quality must be one of %s" % ", ".join(QUALITY_LEVELS))
    cfg = QUALITY_LEVELS[quality]
    c1_tab, c2_tab = _coeff_arrays(coeffs)
    total, n = rows.shape
    x = rows.astype(np.int32)

    per_row = max(1, min(cfg["n_pred"], len(c1_tab)) * len(cfg["mults"]))
    step = max(1, min(_LANES_PER_CHUNK // per_row, _ELEMS_PER_CHUNK // n))

    preds = np.empty(total, dtype=np.uint8)
    deltas = np.empty(total, dtype=np.int32)
    nibs = np.empty((total, n - 2), dtype=np.uint8)
    errs = np.empty(total, dtype=np.float64)
    done = 0
    if progress:
        progress(0, total)
    for lo in range(0, total, step):
        hi = min(total, lo + step)
        p, d, nb, e = _search_rows(
            x[lo:hi], c1_tab, c2_tab, cfg["n_pred"], cfg["mults"], cfg["refine"]
        )
        preds[lo:hi], deltas[lo:hi], nibs[lo:hi], errs[lo:hi] = p, d, nb, e
        done = hi
        if progress:
            progress(done, total)
    return preds, deltas, nibs, errs


def _le16(v: np.ndarray) -> np.ndarray:
    """(B,) int -> (B, 2) little-endian bytes (two's complement)."""
    v = np.asarray(v, dtype=np.int64) & 0xFFFF
    return np.stack([v & 0xFF, v >> 8], axis=1).astype(np.uint8)


def _pack_blocks(
    x: np.ndarray,
    preds: np.ndarray,
    deltas: np.ndarray,
    nibs: np.ndarray,
    channels: int,
    block_align: int,
) -> bytes:
    """Assemble final blocks. ``x`` is (C*B, N); rows are channel-major."""
    total, n = x.shape
    b = total // channels
    out = np.empty((b, block_align), dtype=np.uint8)
    if channels == 1:
        out[:, 0] = preds
        out[:, 1:3] = _le16(deltas)
        out[:, 3:5] = _le16(x[:, 1])
        out[:, 5:7] = _le16(x[:, 0])
        out[:, 7:] = (nibs[:, 0::2] << 4) | nibs[:, 1::2]
    else:
        pl, pr = preds[:b], preds[b:]
        out[:, 0], out[:, 1] = pl, pr
        out[:, 2:4], out[:, 4:6] = _le16(deltas[:b]), _le16(deltas[b:])
        out[:, 6:8], out[:, 8:10] = _le16(x[:b, 1]), _le16(x[b:, 1])
        out[:, 10:12], out[:, 12:14] = _le16(x[:b, 0]), _le16(x[b:, 0])
        out[:, 14:] = (nibs[:b] << 4) | nibs[b:]
    return out.tobytes()


def encode(
    pcm,
    *,
    channels: Optional[int] = None,
    block_align: Optional[int] = None,
    samples_per_block: Optional[int] = None,  # noqa: F811 - public keyword name
    quality: str = "high",
    coeffs: Optional[Coeffs] = None,
    progress: Optional[ProgressFn] = None,
) -> bytes:
    """Encode PCM to raw MS ADPCM blocks.

    Parameters
    ----------
    pcm
        ``int16`` array shaped ``(frames,)``, ``(frames, 1)`` or ``(frames, 2)``;
        floating-point arrays in [-1, 1]; or ``bytes`` of interleaved
        little-endian int16 (then give ``channels``).
    channels
        Only needed for 1-D interleaved stereo or raw bytes.
    block_align, samples_per_block
        Size of one block in bytes, or, alternatively, in sample frames.
        Defaults to ``256 * channels`` bytes.
    quality
        ``"fast"``, ``"normal"`` or ``"high"`` (see :data:`QUALITY_LEVELS`).
    coeffs
        Custom predictor table; default is the seven standard pairs.
    progress
        Optional ``callback(done, total)`` called while encoding.

    Returns
    -------
    bytes
        Concatenated blocks.  The last block is padded by repeating the final
        sample; store the true frame count (e.g. in a WAV ``fact`` chunk) and
        pass it to :func:`decode` as ``n_samples`` to drop the padding.
    """
    frames = _as_frames(pcm, channels)
    n_frames, ch = frames.shape
    block_align, n = _resolve_geometry(ch, block_align, samples_per_block)
    if n_frames == 0:
        return b""

    pad = (-n_frames) % n
    if pad:
        frames = np.concatenate([frames, np.repeat(frames[-1:], pad, axis=0)], axis=0)
    blocks = frames.shape[0] // n
    # (B, N, C) -> channel-major rows (C*B, N)
    rows = frames.reshape(blocks, n, ch).transpose(2, 0, 1).reshape(ch * blocks, n)
    rows = np.ascontiguousarray(rows)

    preds, deltas, nibs, _ = _encode_rows(rows, coeffs, quality, progress)
    return _pack_blocks(rows.astype(np.int32), preds, deltas, nibs, ch, block_align)


# --------------------------------------------------------------------------
# Decoder
# --------------------------------------------------------------------------
def _s16(blocks: np.ndarray, off: int) -> np.ndarray:
    v = blocks[:, off].astype(np.int32) | (blocks[:, off + 1].astype(np.int32) << 8)
    return (v ^ 0x8000) - 0x8000


def _decode_full_blocks(
    blocks: np.ndarray,
    channels: int,
    c1_tab: np.ndarray,
    c2_tab: np.ndarray,
) -> np.ndarray:
    """Decode (B, block_align) uint8 -> (B, N, C) int16."""
    b, ba = blocks.shape
    n = (ba - 7 * channels) * 2 // channels + 2
    if channels == 1:
        pred_idx = blocks[:, 0].astype(np.int64)[None]
        delta = _s16(blocks, 1)[None]
        s1 = _s16(blocks, 3)[None]
        s2 = _s16(blocks, 5)[None]
        data = blocks[:, 7:]
        nib = np.empty((b, n - 2), dtype=np.uint8)
        nib[:, 0::2] = data >> 4
        nib[:, 1::2] = data & 15
        nib = nib[None]  # (1, B, N-2)
    else:
        pred_idx = np.stack([blocks[:, 0], blocks[:, 1]]).astype(np.int64)
        delta = np.stack([_s16(blocks, 2), _s16(blocks, 4)])
        s1 = np.stack([_s16(blocks, 6), _s16(blocks, 8)])
        s2 = np.stack([_s16(blocks, 10), _s16(blocks, 12)])
        data = blocks[:, 14:]
        nib = np.stack([data >> 4, data & 15])  # (2, B, N-2)

    if pred_idx.max(initial=0) >= len(c1_tab):
        raise ValueError("invalid block predictor index %d" % int(pred_idx.max()))
    c1 = c1_tab[pred_idx]
    c2 = c2_tab[pred_idx]

    out = np.empty((channels, b, n), dtype=np.int32)
    out[:, :, 0] = s2
    out[:, :, 1] = s1
    for i in range(2, n):
        u = nib[:, :, i - 2].astype(np.int32)
        pred = _trunc_div256(s1 * c1 + s2 * c2)
        pred = pred + ((u ^ 8) - 8) * delta
        np.clip(pred, -32768, 32767, out=pred)
        out[:, :, i] = pred
        s2 = s1
        s1 = pred
        delta = (_ADAPT[u] * delta) >> 8
        np.maximum(delta, MIN_DELTA, out=delta)
        np.minimum(delta, _DELTA_CEIL, out=delta)
    return out.transpose(1, 2, 0).astype(np.int16)  # (B, N, C)


def decode(
    data,
    *,
    channels: int,
    block_align: int,
    n_samples: Optional[int] = None,
    coeffs: Optional[Coeffs] = None,
) -> np.ndarray:
    """Decode raw MS ADPCM blocks to an ``int16`` array of shape ``(frames, channels)``.

    A trailing partial block (data length not a multiple of ``block_align``)
    is decoded as far as it goes.  ``n_samples`` (e.g. from a ``fact`` chunk)
    truncates the result to the real length.
    """
    _check_channels(channels)
    n = samples_per_block(block_align, channels)
    raw = np.frombuffer(bytes(data) if not isinstance(data, (bytes, bytearray)) else data,
                        dtype=np.uint8)
    c1_tab, c2_tab = _coeff_arrays(coeffs)

    full = raw.size // block_align
    tail = raw.size - full * block_align
    pieces = []
    if full:
        pieces.append(
            _decode_full_blocks(raw[: full * block_align].reshape(full, block_align),
                                channels, c1_tab, c2_tab).reshape(-1, channels)
        )
    hdr = _header_size(channels)
    if tail > hdr:
        padded = np.zeros((1, block_align), dtype=np.uint8)
        padded[0, :tail] = raw[full * block_align:]
        frames = _decode_full_blocks(padded, channels, c1_tab, c2_tab).reshape(-1, channels)
        valid = 2 + (tail - hdr) * 2 // channels
        pieces.append(frames[:valid])
    out = np.concatenate(pieces, axis=0) if pieces else np.zeros((0, channels), np.int16)
    if n_samples is not None:
        out = out[: max(0, int(n_samples))]
    assert n >= 2
    return out


# --------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------
def snr_db(reference, test) -> float:
    """Signal-to-noise ratio in dB between two int16 arrays (inf if identical)."""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(test, dtype=np.float64)
    k = min(len(a), len(b))
    a, b = a[:k], b[:k]
    noise = float(np.sum((a - b) ** 2))
    sig = float(np.sum(a ** 2))
    if noise == 0.0:
        return float("inf")
    if sig == 0.0:
        return float("-inf")
    return 10.0 * np.log10(sig / noise)
