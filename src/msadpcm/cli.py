"""Command line interface and the file-conversion helper shared with the GUI demo."""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from . import codec, wav

__all__ = ["ConversionResult", "convert_file", "main"]


@dataclass
class ConversionResult:
    """What :func:`convert_file` did."""

    src: Path
    dst: Path
    mode: str  # "encode" or "decode"
    frames: int
    sample_rate: int
    channels: int
    src_bytes: int
    dst_bytes: int
    seconds: float
    block_align: Optional[int] = None
    samples_per_block: Optional[int] = None
    snr_db: Optional[float] = None  # encode only: ADPCM output vs. the input

    @property
    def ratio(self) -> float:
        return self.src_bytes / self.dst_bytes if self.dst_bytes else float("inf")


def convert_file(
    src,
    dst=None,
    *,
    mode: str = "encode",
    block_align: Optional[int] = None,
    samples_per_block: Optional[int] = None,
    quality: str = "normal",
    verify: bool = True,
    progress: Optional[Callable[[int, int], None]] = None,
) -> ConversionResult:
    """Convert one WAV file.

    ``mode="encode"`` writes MS ADPCM, ``mode="decode"`` writes 16-bit PCM.
    With ``verify`` the encoded file is decoded again to report the SNR.
    """
    if mode not in ("encode", "decode"):
        raise ValueError("mode must be 'encode' or 'decode'")
    src = Path(src)
    if dst is None:
        dst = src.with_name(src.stem + (".adpcm.wav" if mode == "encode" else ".pcm.wav"))
    dst = Path(dst)

    t0 = time.perf_counter()
    pcm, rate = wav.read_wav(src)
    frames, ch = pcm.shape

    if mode == "decode":
        written = wav.write_wav(dst, pcm, rate, adpcm=False)
        return ConversionResult(src, dst, mode, frames, rate, ch, src.stat().st_size,
                                written, time.perf_counter() - t0)

    if block_align is None and samples_per_block is None:
        block_align = codec.default_block_align(rate, ch)
    ba, spb = codec._resolve_geometry(ch, block_align, samples_per_block)
    written = wav.write_wav(dst, pcm, rate, block_align=ba, quality=quality, progress=progress)
    snr = None
    if verify:
        decoded, _ = wav.read_wav(dst)
        snr = codec.snr_db(pcm, decoded)
    return ConversionResult(src, dst, mode, frames, rate, ch, src.stat().st_size, written,
                            time.perf_counter() - t0, ba, spb, snr)


class _ProgressLine:
    """Single updating status line on stderr (``name: encoding 45%``).

    Shown only on an interactive terminal, so redirected output and logs stay
    clean.  Call the instance as the ``progress`` callback; call :meth:`clear`
    before printing anything else.
    """

    def __init__(self, label: str, stream=None) -> None:
        self.label = label
        self.stream = sys.stderr if stream is None else stream
        self.enabled = bool(getattr(self.stream, "isatty", lambda: False)())
        self._shown = 0
        self._last = -1
        self._t0 = time.perf_counter()

    def __call__(self, done: int, total: int) -> None:
        if not self.enabled or total <= 0:
            return
        percent = 100 * done // total
        if percent == self._last:
            return
        self._last = percent
        elapsed = time.perf_counter() - self._t0
        text = "%s: encoding %d%%" % (self.label, percent)
        if 0 < done < total and elapsed >= 2.0:
            text += "  (%ds left)" % round(elapsed * (total - done) / done)
        self._write(text)

    def _write(self, text: str) -> None:
        pad = " " * max(0, self._shown - len(text))
        self.stream.write("\r" + text + pad)
        self.stream.flush()
        self._shown = len(text)

    def clear(self) -> None:
        if self.enabled and self._shown:
            self.stream.write("\r" + " " * self._shown + "\r")
            self.stream.flush()
            self._shown = 0


def _describe(r: ConversionResult) -> str:
    text = "%s -> %s  %.1fs audio, %s, %.2fs" % (
        r.src.name, r.dst.name, r.frames / r.sample_rate,
        "stereo" if r.channels == 2 else "mono", r.seconds)
    text += "  [%d -> %d bytes, x%.2f]" % (r.src_bytes, r.dst_bytes, r.ratio)
    if r.mode == "encode":
        text += "  block %d bytes (%d samples)" % (r.block_align, r.samples_per_block)
        if r.snr_db is not None:
            text += "  SNR %.2f dB" % r.snr_db
    return text


def _targets(inputs: Sequence[str], out: Optional[str], outdir: Optional[str], mode: str):
    if out and len(inputs) != 1:
        raise SystemExit("-o/--output works with exactly one input file; use -d/--outdir")
    suffix = ".adpcm.wav" if mode == "encode" else ".pcm.wav"
    for item in inputs:
        src = Path(item)
        if out:
            dst = Path(out)
        elif outdir:
            dst = Path(outdir) / (src.stem + suffix)
        else:
            dst = src.with_name(src.stem + suffix)
        yield src, dst


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="msadpcm", description="Microsoft ADPCM WAV encoder/decoder.")
    sub = p.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode", help="PCM WAV -> MS ADPCM WAV")
    enc.add_argument("inputs", nargs="+")
    g = enc.add_mutually_exclusive_group()
    g.add_argument("--block-align", type=int, help="bytes per block (default: like Windows ACM)")
    g.add_argument("--samples-per-block", type=int, help="sample frames per block, e.g. 128")
    enc.add_argument("--quality", choices=list(codec.QUALITY_LEVELS), default="normal")
    enc.add_argument("-o", "--output")
    enc.add_argument("-d", "--outdir")
    enc.add_argument("--no-verify", action="store_true", help="skip the SNR check")

    dec = sub.add_parser("decode", help="MS ADPCM WAV -> 16-bit PCM WAV")
    dec.add_argument("inputs", nargs="+")
    dec.add_argument("-o", "--output")
    dec.add_argument("-d", "--outdir")

    info = sub.add_parser("info", help="show WAV header information")
    info.add_argument("inputs", nargs="+")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "info":
        for item in args.inputs:
            i = wav.wav_info(item)
            kind = {1: "PCM", 2: "MS ADPCM", 3: "float", 0xFFFE: "extensible"}.get(
                i.format_tag, "0x%04X" % i.format_tag)
            line = "%s: %s, %d ch, %d Hz, %d-bit" % (
                item, kind, i.channels, i.sample_rate, i.bits_per_sample)
            if i.is_adpcm:
                line += ", block %d bytes (%d samples)" % (
                    i.block_align, codec.samples_per_block(i.block_align, i.channels))
            if i.n_frames is not None:
                line += ", %d frames (%.2fs)" % (i.n_frames, i.duration)
            print(line)
        return 0

    failures = 0
    for src, dst in _targets(args.inputs, args.output, args.outdir, args.cmd):
        try:
            if args.outdir:
                Path(args.outdir).mkdir(parents=True, exist_ok=True)
            kwargs = {}
            line = None
            if args.cmd == "encode":
                line = _ProgressLine(src.name)
                kwargs = dict(block_align=args.block_align,
                              samples_per_block=args.samples_per_block,
                              quality=args.quality, verify=not args.no_verify,
                              progress=line)
            try:
                result = convert_file(src, dst, mode=args.cmd, **kwargs)
            finally:
                if line is not None:
                    line.clear()
            print(_describe(result))
        except Exception as exc:  # report and continue with the next file
            failures += 1
            print("%s: error: %s" % (src, exc), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
