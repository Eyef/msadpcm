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

import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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
# The initial delta of a block has a large and very irregular effect on its
# error: the delta adapts after every sample, and over a few dozen samples the
# result depends chaotically on where it started.  Measured on real recordings
# with 128-sample blocks, the best delta among ~100 values spaced 3 % apart is
# often several dB better than the best of a coarse grid, and the dense curve
# has many narrow dips.  A finer search therefore pays off, but only for one
# predictor: the predictors differ so much (a median of 10 to 30 dB on most
# material) that a coarse pass picks the right one reliably.  So the search is
#
#   1. coarse:  every predictor x a few deltas around an estimate; keep the best ``keep``
#               predictors,
#   2. dense:   for those, every delta of a fixed ~3 %-spaced integer grid inside a
#               window around the estimate (every ``step``-th grid point),
#   3. polish:  deltas just next to the winner (only when ``step`` > 1),
#   4. one last pass that records the nibbles of the winner.
#
#: ``n_pred``  - predictors entering the coarse stage (all seven standard ones; a
#:               custom table with more pairs is pre-ranked by open-loop error)
#: ``coarse``  - initial deltas of the coarse stage, as multiples of the estimate
#:               (mean |open-loop residual| of the first 16 samples)
#: ``keep``    - predictors that get the dense search (0: none, the coarse winner is used)
#: ``step``    - dense grid stride; 1 means every point (spacing 3 %), 4 about 13 %
#: ``window``  - dense search range, as multiples of the estimate
def _grid(*parts) -> Tuple[float, ...]:
    return tuple(sorted({round(float(v), 4) for p in parts for v in p}))


_GRID_RATIO = 1.03


def _make_delta_grid() -> np.ndarray:
    k = np.arange(0, 400)
    g = np.unique(np.rint(MIN_DELTA * _GRID_RATIO ** k).astype(np.int64))
    return g[g <= MAX_HEADER_DELTA].astype(np.int32)


#: Distinct integer deltas, spaced about 3 % apart (all integers up to 33).
_DELTA_GRID = _make_delta_grid()

_FAST = (0.3, 0.7, 1.4)
_COARSE5 = (0.12, 0.22, 0.4, 0.7, 1.2)
_COARSE7 = (0.09, 0.15, 0.25, 0.4, 0.65, 1.05, 1.7)

# Roughly 21 / 67 / 140 / 330 candidate lanes per block for a 128-sample block.
QUALITY_LEVELS = {
    "fast":   dict(n_pred=7, coarse=_FAST,    keep=0, step=0, window=(0.08, 1.6)),
    "normal": dict(n_pred=7, coarse=_COARSE5, keep=1, step=4, window=(0.08, 1.6)),
    "high":   dict(n_pred=7, coarse=_COARSE7, keep=1, step=1, window=(0.08, 1.6)),
    "max":    dict(n_pred=7, coarse=_COARSE7, keep=2, step=1, window=(0.05, 2.0)),
}

#: Rough number of candidate lanes processed per vectorised chunk.  Around 2**15 the
#: working arrays stay in the CPU cache; 2**17 was about 20 % slower, and much smaller
#: chunks lose to the per-call overhead of NumPy.  Progress reporting does not depend on it.
_LANES_PER_CHUNK = 1 << 15
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
    lane_row: np.ndarray,
    lane_pred: np.ndarray,
    lane_delta: np.ndarray,
    c1_tab: np.ndarray,
    c2_tab: np.ndarray,
    want_nibs: bool = False,
    tick: Optional[Callable[[float], None]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Encode block ``lane_row[k]`` with predictor ``lane_pred[k]`` and initial
    delta ``lane_delta[k]``, for every lane ``k`` at once.

    x      : (r, N) int32 samples
    want_nibs : also return the nibbles; only worth it for few lanes, the buffer
                is (N-2) * lanes bytes
    tick   : optional ``callback(fraction_done)`` called every 32 samples
    returns: err (lanes,) float64 squared error, nibbles (N-2, lanes) uint8 or None
    """
    n = x.shape[1]
    lanes = len(lane_row)
    xt = np.ascontiguousarray(x.T)  # (N, r)

    c1 = c1_tab[lane_pred]
    c2 = c2_tab[lane_pred]
    s1 = xt[1][lane_row].astype(np.int32)
    s2 = xt[0][lane_row].astype(np.int32)
    delta = np.asarray(lane_delta).astype(np.int32)
    err = np.zeros(lanes, dtype=np.float64)
    over = np.zeros(lanes, dtype=bool)
    nibs = np.empty((n - 2, lanes), dtype=np.uint8) if want_nibs else None

    for i in range(2, n):
        if tick is not None and not (i & 31):
            tick((i - 2) / (n - 2))
        pred = _trunc_div256(s1 * c1 + s2 * c2)
        target = xt[i][lane_row]
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
    return err, nibs


def _argmin_runs(err: np.ndarray, counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row minimum of ``err`` whose rows are consecutive runs of ``counts``
    (all >= 1) lanes.  Returns (index of the first minimum, minimum)."""
    starts = np.zeros(len(counts), dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    mins = np.minimum.reduceat(err, starts)
    pos = np.flatnonzero(err == np.repeat(mins, counts))
    row_of = np.repeat(np.arange(len(counts)), counts)[pos]
    first = pos[np.searchsorted(row_of, np.arange(len(counts)))]
    return first, mins


def _search_rows(
    x: np.ndarray,
    c1_tab: np.ndarray,
    c2_tab: np.ndarray,
    cfg: dict,
    tick: Optional[Callable[[float], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find predictor, initial delta and nibbles for each row of ``x``.

    ``tick(fraction)`` reports progress over the simulation passes, weighted by
    the number of candidate lanes each one processes (the dense pass's weight is
    an estimate, so the fraction may jump a little when it ends).
    """
    r, n = x.shape
    n_coef = len(c1_tab)
    n_pred = min(cfg["n_pred"], n_coef)
    coarse = np.asarray(cfg["coarse"], dtype=np.float64)
    keep = min(cfg["keep"], n_pred)
    step = cfg["step"]
    lo_w, hi_w = cfg["window"]
    ar = np.arange(r)

    # Which predictors enter the coarse stage.
    if n_coef > n_pred:  # only for custom tables with many pairs
        score = np.empty((n_coef, r), dtype=np.float64)
        for p in range(n_coef):
            rp = (x[:, 2:] - _trunc_div256(x[:, 1:-1] * c1_tab[p] + x[:, :-2] * c2_tab[p]))
            rp = rp.astype(np.float64)
            score[p] = np.einsum("ij,ij->i", rp, rp)
        sel = np.argsort(score, axis=0)[:n_pred].T  # (r, n_pred) best first
    else:
        sel = np.broadcast_to(np.arange(n_coef), (r, n_coef))

    # Delta estimate per (row, predictor): scale of the first few open-loop residuals.
    head = min(16, n - 2)
    est = np.empty((r, n_pred), dtype=np.float64)
    for j in range(n_pred):
        p = sel[:, j]
        h = x[:, 2:2 + head] - _trunc_div256(
            x[:, 1:1 + head] * c1_tab[p][:, None] + x[:, :head] * c2_tab[p][:, None])
        est[:, j] = np.abs(h).astype(np.float64).mean(axis=1)
    est = np.maximum(est, MIN_DELTA)

    # ---- stage 1: coarse ------------------------------------------------
    m1 = len(coarse)
    d1 = np.clip(np.rint(est[:, :, None] * coarse[None, None, :]), MIN_DELTA, MAX_HEADER_DELTA)
    d1 = d1.astype(np.int32).reshape(r, -1)
    p1 = np.repeat(sel, m1, axis=1)  # (r, n_pred * m1)
    lanes1 = n_pred * m1

    # Dense-stage size, known from the estimates alone (upper bound for the weights).
    dense = keep > 0 and step > 0
    if dense:
        lo_i = np.searchsorted(_DELTA_GRID, est * lo_w, side="left")
        hi_i = np.searchsorted(_DELTA_GRID, est * hi_w, side="right") - 1
        hi_i = np.maximum(hi_i, lo_i)
        hi_i = np.minimum(hi_i, len(_DELTA_GRID) - 1)
        lo_i = np.minimum(lo_i, hi_i)
        cnt_all = (hi_i - lo_i) // step + 1  # (r, n_pred)
        w_dense = keep * float(cnt_all.mean())
    else:
        w_dense = 0.0
    w_pol = 2.0 * (step - 1) if dense and step > 1 else 0.0
    w_all = lanes1 + w_dense + w_pol + 1.0

    def stage(lo: float, hi: float):
        if tick is None:
            return None
        return lambda f: tick((lo + f * (hi - lo)) / w_all)

    err1, _ = _simulate(
        x, np.repeat(ar, lanes1), p1.reshape(-1), d1.reshape(-1), c1_tab, c2_tab,
        tick=stage(0, lanes1))
    err1 = err1.reshape(r, n_pred, m1)
    k1 = err1.argmin(axis=2)  # (r, n_pred)
    e1 = np.take_along_axis(err1, k1[:, :, None], axis=2)[:, :, 0]
    dd1 = np.take_along_axis(d1.reshape(r, n_pred, m1), k1[:, :, None], axis=2)[:, :, 0]
    order = np.argsort(e1, axis=1)  # (r, n_pred) best first
    top = order[:, 0]
    best_pred = sel[ar, top]
    best_delta = dd1[ar, top]
    best_err = e1[ar, top]

    # ---- stage 2: dense -------------------------------------------------
    if dense:
        cols = order[:, :keep]  # (r, keep)
        lo_k = np.take_along_axis(lo_i, cols, axis=1)
        cnt_k = np.take_along_axis(cnt_all, cols, axis=1)
        pred_k = np.take_along_axis(sel, cols, axis=1)
        counts = cnt_k.reshape(-1)  # runs: (row, kept predictor), row-major
        total = int(counts.sum())
        run_start = np.zeros(len(counts), dtype=np.int64)
        np.cumsum(counts[:-1], out=run_start[1:])
        within = np.arange(total) - np.repeat(run_start, counts)
        gidx = np.repeat(lo_k.reshape(-1), counts) + step * within
        lane_delta = _DELTA_GRID[np.minimum(gidx, len(_DELTA_GRID) - 1)]
        lane_pred = np.repeat(pred_k.reshape(-1), counts)
        lane_row = np.repeat(np.repeat(ar, keep), counts)
        err2, _ = _simulate(
            x, lane_row, lane_pred, lane_delta, c1_tab, c2_tab, tick=stage(lanes1, lanes1 + w_dense))
        row_counts = cnt_k.sum(axis=1)
        first, mins = _argmin_runs(err2, row_counts)
        better = mins < best_err
        best_pred = np.where(better, lane_pred[first], best_pred)
        best_delta = np.where(better, lane_delta[first], best_delta)
        best_err = np.where(better, mins, best_err)

        # ---- stage 3: polish, a few deltas next to the winner ----------
        if step > 1:
            j = np.concatenate([np.arange(-(step - 1), 0), np.arange(1, step)])
            pd = np.rint(best_delta[:, None] * _GRID_RATIO ** j[None, :])
            pd = np.clip(pd, MIN_DELTA, MAX_HEADER_DELTA).astype(np.int32)
            m3 = pd.shape[1]
            err3, _ = _simulate(
                x, np.repeat(ar, m3), np.repeat(best_pred, m3), pd.reshape(-1), c1_tab, c2_tab,
                tick=stage(lanes1 + w_dense, lanes1 + w_dense + w_pol))
            err3 = err3.reshape(r, m3)
            b3 = err3.argmin(axis=1)
            e3 = err3[ar, b3]
            better = e3 < best_err
            best_delta = np.where(better, pd[ar, b3], best_delta)
            best_err = np.where(better, e3, best_err)

    # Only the winners need their nibbles: one extra pass over r lanes.
    final_err, best_nibs = _simulate(
        x, ar, best_pred, best_delta, c1_tab, c2_tab, want_nibs=True,
        tick=stage(w_all - 1.0, w_all))
    return (
        best_pred.astype(np.uint8),
        best_delta.astype(np.int32),
        np.ascontiguousarray(best_nibs.T, dtype=np.uint8),
        final_err,
    )


def _search_chunk(args):
    """Worker entry point (module level so that it can be pickled)."""
    x, c1_tab, c2_tab, cfg = args
    return _search_rows(x, c1_tab, c2_tab, cfg)


def _resolve_workers(workers: Optional[int]) -> int:
    if workers is None or workers == 0:
        return os.cpu_count() or 1
    if workers < 0 or int(workers) != workers:
        raise ValueError("workers must be a positive integer, or 0 for all CPUs")
    return int(workers)


#: Fewest rows worth sending to a worker process in one piece.
_MIN_ROWS_PER_TASK = 64
#: Below this much work (samples x candidate lanes, roughly 0.8 s of single-process
#: time) starting worker processes costs more than it saves.
_MIN_PARALLEL_WORK = 5e7


def _encode_rows(
    rows: np.ndarray,
    coeffs: Optional[Coeffs],
    quality: str,
    progress: Optional[ProgressFn],
    workers: Optional[int] = 1,
):
    """Encode (R, N) int16 rows. Returns preds, deltas, nibbles, errors.

    Every row is decided on its own, so the result does not depend on how the
    rows are split into chunks or on ``workers``.
    """
    if quality not in QUALITY_LEVELS:
        raise ValueError("quality must be one of %s" % ", ".join(QUALITY_LEVELS))
    workers = _resolve_workers(workers)
    cfg = QUALITY_LEVELS[quality]
    c1_tab, c2_tab = _coeff_arrays(coeffs)
    total, n = rows.shape
    x = rows.astype(np.int32)

    n_pred = min(cfg["n_pred"], len(c1_tab))
    per_row = n_pred * len(cfg["coarse"])
    if cfg["keep"] > 0 and cfg["step"] > 0:
        # dense stage upper bound: the whole window of the grid, every ``step``-th point
        span = np.log(cfg["window"][1] / cfg["window"][0]) / np.log(_GRID_RATIO)
        per_row = max(per_row, cfg["keep"] * int(span / cfg["step"] + 2))
    step = max(1, min(_LANES_PER_CHUNK // max(1, per_row), _ELEMS_PER_CHUNK // n))
    if total * n * per_row < _MIN_PARALLEL_WORK:
        workers = 1  # decided first: the smaller chunks below are only good for workers
    if workers > 1:  # several chunks per worker keep all of them busy until the end
        step = max(min(step, _MIN_ROWS_PER_TASK), min(step, -(-total // (workers * 4))))
    spans = [(lo, min(total, lo + step)) for lo in range(0, total, step)]
    if len(spans) < 2:
        workers = 1

    preds = np.empty(total, dtype=np.uint8)
    deltas = np.empty(total, dtype=np.int32)
    nibs = np.empty((total, n - 2), dtype=np.uint8)
    errs = np.empty(total, dtype=np.float64)
    last = -1

    def report(done: int) -> None:
        nonlocal last
        if progress and done > last:
            last = done
            progress(done, total)

    def store(lo, hi, res):
        preds[lo:hi], deltas[lo:hi], nibs[lo:hi], errs[lo:hi] = res

    report(0)
    if workers == 1:
        for lo, hi in spans:
            tick = (lambda f, lo=lo, hi=hi: report(lo + int(f * (hi - lo)))) if progress else None
            store(lo, hi, _search_rows(x[lo:hi], c1_tab, c2_tab, cfg, tick))
            report(hi)
        return preds, deltas, nibs, errs

    done_rows = 0
    pending = {}
    queue = iter(spans)
    ex = ProcessPoolExecutor(max_workers=workers)
    try:
        while True:
            while len(pending) < 2 * workers:  # bounded: only a few chunks are pickled at a time
                span = next(queue, None)
                if span is None:
                    break
                lo, hi = span
                pending[ex.submit(_search_chunk, (x[lo:hi], c1_tab, c2_tab, cfg))] = span
            if not pending:
                break
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                lo, hi = pending.pop(fut)
                store(lo, hi, fut.result())
                done_rows += hi - lo
                report(done_rows)
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
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
    quality: str = "normal",
    coeffs: Optional[Coeffs] = None,
    progress: Optional[ProgressFn] = None,
    workers: Optional[int] = 1,
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
        ``"fast"``, ``"normal"`` (default), ``"high"`` or ``"max"``; slower presets
        try more candidates and gain a fraction of a dB (see :data:`QUALITY_LEVELS`).
    coeffs
        Custom predictor table; default is the seven standard pairs.  Files made
        with a different table are not portable: FFmpeg's decoder ignores or
        rejects it (see the README), so use this only with your own decoder.
    progress
        Optional ``callback(done, total)`` called while encoding.
    workers
        Number of worker processes (default 1: everything runs in the calling
        process).  ``0`` means one per CPU.  The output is byte-for-byte the same
        for any value.  Small inputs (about a second of work) are encoded in the
        calling process regardless, since starting workers would cost more; on
        Windows and macOS the workers are started with *spawn*, so a script that
        calls this must be guarded by ``if __name__ == "__main__":``.

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

    preds, deltas, nibs, _ = _encode_rows(rows, coeffs, quality, progress, workers)
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
