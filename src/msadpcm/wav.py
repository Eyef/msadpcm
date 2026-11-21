"""Minimal RIFF/WAVE reading and writing for PCM and MS ADPCM files.

The standard library's :mod:`wave` cannot open ADPCM files, so this module
parses the chunks itself.  It understands:

* ``WAVE_FORMAT_PCM`` (8/16/24/32-bit integer), ``WAVE_FORMAT_IEEE_FLOAT``
  and ``WAVE_FORMAT_EXTENSIBLE`` wrappers of those, returned as int16;
* ``WAVE_FORMAT_ADPCM`` (MS ADPCM), decoded with :func:`msadpcm.decode`,
  including a custom coefficient table stored in the ``fmt `` chunk.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from . import codec

__all__ = ["WavInfo", "format_chunk", "read_wav", "read_adpcm_raw", "write_wav", "wav_info"]

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_ADPCM = 0x0002
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE

PathLike = Union[str, Path]


@dataclass
class WavInfo:
    """Header information of a WAV file."""

    format_tag: int
    channels: int
    sample_rate: int
    block_align: int
    bits_per_sample: int
    n_frames: Optional[int]
    coeffs: Optional[Tuple[Tuple[int, int], ...]] = None

    @property
    def is_adpcm(self) -> bool:
        return self.format_tag == WAVE_FORMAT_ADPCM

    @property
    def duration(self) -> Optional[float]:
        return None if self.n_frames is None else self.n_frames / self.sample_rate


# --------------------------------------------------------------------------
# fmt chunk
# --------------------------------------------------------------------------
def format_chunk(
    sample_rate: int,
    channels: int,
    block_align: int,
    coeffs: Optional[codec.Coeffs] = None,
) -> bytes:
    """Body of the ``fmt `` chunk (WAVEFORMATEX + ADPCM extra data), 50 bytes
    for the standard seven coefficient pairs."""
    spb = codec.samples_per_block(block_align, channels)
    table = codec.STANDARD_COEFFS if coeffs is None else tuple(coeffs)
    extra = struct.pack("<HH", spb, len(table))
    for c1, c2 in table:
        extra += struct.pack("<hh", c1, c2)
    avg_bytes = (sample_rate * block_align) // spb
    return struct.pack(
        "<HHIIHHH", WAVE_FORMAT_ADPCM, channels, sample_rate, avg_bytes,
        block_align, 4, len(extra),
    ) + extra


def _parse_fmt(body: bytes) -> WavInfo:
    if len(body) < 16:
        raise ValueError("fmt chunk too short")
    tag, ch, rate, _avg, align, bits = struct.unpack_from("<HHIIHH", body, 0)
    coeffs = None
    if tag == WAVE_FORMAT_EXTENSIBLE and len(body) >= 26:
        # sub-format GUID starts with the real format tag
        tag = struct.unpack_from("<H", body, 24)[0]
    if tag == WAVE_FORMAT_ADPCM and len(body) >= 22:
        _spb, ncoef = struct.unpack_from("<HH", body, 18)
        if 0 < ncoef <= 256 and len(body) >= 22 + 4 * ncoef:
            vals = struct.unpack_from("<%dh" % (2 * ncoef), body, 22)
            coeffs = tuple((vals[2 * i], vals[2 * i + 1]) for i in range(ncoef))
    return WavInfo(tag, ch, rate, align, bits, None, coeffs)


# --------------------------------------------------------------------------
# chunk walking
# --------------------------------------------------------------------------
def _read_chunks(raw: bytes) -> List[Tuple[bytes, int, int]]:
    """Return ``(id, body_offset, body_size)`` for every top-level chunk."""
    if len(raw) < 12 or raw[:4] not in (b"RIFF", b"RIFX") or raw[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    chunks = []
    pos = 12
    while pos + 8 <= len(raw):
        cid, size = struct.unpack_from("<4sI", raw, pos)
        body = pos + 8
        if cid == b"data" and body + size > len(raw):
            size = len(raw) - body  # streamed / truncated file
        size = min(size, len(raw) - body)
        chunks.append((cid, body, size))
        pos = body + size + (size & 1)
    return chunks


def _parse(raw: bytes):
    info = None
    fact = None
    data_span = None
    for cid, off, size in _read_chunks(raw):
        if cid == b"fmt " and info is None:
            info = _parse_fmt(raw[off:off + size])
        elif cid == b"fact" and size >= 4:
            fact = struct.unpack_from("<I", raw, off)[0]
        elif cid == b"data" and data_span is None:
            data_span = (off, size)
    if info is None:
        raise ValueError("missing fmt chunk")
    if data_span is None:
        raise ValueError("missing data chunk")
    if info.is_adpcm:
        info.n_frames = fact
    return info, data_span


def wav_info(path: PathLike) -> WavInfo:
    """Read header information (and frame count) without decoding.

    For MS ADPCM the frame count from the ``fact`` chunk is checked against what
    the ``data`` chunk can actually hold: some tools write a ``fact`` value that
    is too large (for stereo, twice the real length), and the smaller number
    is the true one.
    """
    raw = Path(path).read_bytes()
    info, (_, size) = _parse(raw)
    if info.is_adpcm:
        ba = info.block_align
        full, tail = divmod(size, ba)
        derived = full * codec.samples_per_block(ba, info.channels)
        hdr = 7 * info.channels
        if tail > hdr:
            derived += 2 + (tail - hdr) * 2 // info.channels
        info.n_frames = derived if info.n_frames is None else min(info.n_frames, derived)
    elif info.block_align:
        info.n_frames = size // info.block_align
    return info


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def _pcm_to_int16(buf: bytes, tag: int, bits: int, channels: int) -> np.ndarray:
    if tag == WAVE_FORMAT_IEEE_FLOAT:
        if bits not in (32, 64):
            raise ValueError("unsupported float width %d" % bits)
        a = np.frombuffer(buf[: len(buf) // (bits // 8) * (bits // 8)],
                          dtype="<f%d" % (bits // 8))
        a = np.clip(np.rint(a.astype(np.float64) * 32768.0), -32768, 32767).astype(np.int16)
    elif tag == WAVE_FORMAT_PCM:
        if bits == 16:
            a = np.frombuffer(buf[: len(buf) // 2 * 2], dtype="<i2").astype(np.int16)
        elif bits == 8:
            a = ((np.frombuffer(buf, dtype=np.uint8).astype(np.int16) - 128) << 8).astype(np.int16)
        elif bits == 24:
            m = len(buf) // 3 * 3
            b = np.frombuffer(buf[:m], dtype=np.uint8).reshape(-1, 3)
            a = (b[:, 1].astype(np.int16) | (b[:, 2].astype(np.int16) << 8)).astype(np.int16)
        elif bits == 32:
            a = (np.frombuffer(buf[: len(buf) // 4 * 4], dtype="<i4") >> 16).astype(np.int16)
        else:
            raise ValueError("unsupported PCM width %d" % bits)
    else:
        raise ValueError("unsupported WAV format tag 0x%04X" % tag)
    usable = a.size // channels * channels
    return a[:usable].reshape(-1, channels)


def read_wav(path: PathLike) -> Tuple[np.ndarray, int]:
    """Read a PCM or MS ADPCM WAV file.

    Returns ``(samples, sample_rate)`` with ``samples`` an int16 array of shape
    ``(frames, channels)``.
    """
    raw = Path(path).read_bytes()
    info, (off, size) = _parse(raw)
    body = raw[off:off + size]
    if info.is_adpcm:
        pcm = codec.decode(
            body, channels=info.channels, block_align=info.block_align,
            n_samples=info.n_frames, coeffs=info.coeffs,
        )
    else:
        pcm = _pcm_to_int16(body, info.format_tag, info.bits_per_sample, info.channels)
    return pcm, info.sample_rate


def read_adpcm_raw(path: PathLike) -> Tuple[bytes, WavInfo]:
    """Return the undecoded ADPCM block data and header of an ADPCM WAV file."""
    raw = Path(path).read_bytes()
    info, (off, size) = _parse(raw)
    if not info.is_adpcm:
        raise ValueError("not an MS ADPCM file")
    return raw[off:off + size], info


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------
def write_wav(
    path: PathLike,
    pcm,
    sample_rate: int,
    *,
    adpcm: bool = True,
    channels: Optional[int] = None,
    block_align: Optional[int] = None,
    samples_per_block: Optional[int] = None,
    quality: str = "normal",
    coeffs: Optional[codec.Coeffs] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Write ``pcm`` to ``path`` as MS ADPCM (or plain 16-bit PCM if ``adpcm=False``).

    Returns the number of bytes written.  When no block size is given it uses
    :func:`msadpcm.default_block_align` for ``sample_rate``.
    """
    frames = codec._as_frames(pcm, channels)
    n_frames, ch = frames.shape

    if not adpcm:
        data = frames.astype("<i2").tobytes()
        fmt = struct.pack("<HHIIHH", WAVE_FORMAT_PCM, ch, sample_rate,
                          sample_rate * ch * 2, ch * 2, 16)
        chunks = [(b"fmt ", fmt), (b"data", data)]
    else:
        if block_align is None and samples_per_block is None:
            block_align = codec.default_block_align(sample_rate, ch)
        ba, _ = codec._resolve_geometry(ch, block_align, samples_per_block)
        data = codec.encode(
            frames, block_align=ba, quality=quality, coeffs=coeffs, progress=progress,
        )
        chunks = [
            (b"fmt ", format_chunk(sample_rate, ch, ba, coeffs)),
            (b"fact", struct.pack("<I", n_frames)),
            (b"data", data),
        ]

    out = bytearray(b"RIFF\0\0\0\0WAVE")
    for cid, body in chunks:
        out += struct.pack("<4sI", cid, len(body)) + body
        if len(body) & 1:
            out += b"\0"
    struct.pack_into("<I", out, 4, len(out) - 8)
    Path(path).write_bytes(bytes(out))
    return len(out)
