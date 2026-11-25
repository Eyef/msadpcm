# msadpcm

[![tests](https://github.com/Eyef/msadpcm/actions/workflows/tests.yml/badge.svg)](https://github.com/Eyef/msadpcm/actions/workflows/tests.yml)

Microsoft ADPCM (`WAVE_FORMAT_ADPCM`, tag `0x0002`) **encoder and decoder** in pure
Python + NumPy. No external programs and no compiled extensions.

* any block size (`block_align` in bytes, or `samples_per_block`), mono and stereo
* decoder output is **bit-exact with FFmpeg's `adpcm_ms`** (checked in the test suite, both directions)
* encoder searches the 7 standard predictors and a fine grid of initial step sizes for every block, with
  the same arithmetic as the decoder, so the quality it measures is the quality you get
* optional multi-process encoding (`workers=`), with byte-for-byte the same output
* reads and writes complete WAV files (`fact` chunk included), or works on raw blocks in memory
* optional custom coefficient tables (see the compatibility warning below)
* command line tool and a small tkinter demo (`examples/gui_demo.py`)

```
pip install msadpcm
```

## Quick start

```python
import numpy as np
import msadpcm

# whole files
msadpcm.write_wav("out.wav", pcm, 22050)                 # pcm: int16 (frames,) or (frames, 2)
pcm, rate = msadpcm.read_wav("out.wav")                  # ADPCM or PCM in, int16 (frames, channels) out

# raw blocks
blocks = msadpcm.encode(pcm, block_align=512)            # quality="normal" by default
pcm = msadpcm.decode(blocks, channels=1, block_align=512, n_samples=len(pcm))
```

Inputs may be `int16` arrays, floats in [-1, 1], plain lists of ints, or little-endian `bytes`
(then pass `channels=`). Wider integer arrays are rejected on purpose: a 24-bit file stored in
`int32` would be silently clipped otherwise; convert it explicitly.

### Block size

```python
msadpcm.write_wav("a.wav", pcm, 44100, block_align=2048)        # bytes per block
msadpcm.write_wav("b.wav", pcm, 22050, samples_per_block=128)   # 70-byte mono / 140-byte stereo blocks
msadpcm.samples_per_block(512, 2)                               # -> 500
msadpcm.block_align_for(128, 1)                                 # -> 70
```

Without an explicit size `write_wav` uses what Windows' ACM codec does
(`256 * channels * max(1, rate // 11025)` bytes). A larger block means a smaller file
(less header overhead) but slower adaptation after a transient.
`samples_per_block=128` gives the 70-byte mono / 140-byte stereo blocks that some game engines
expect (for example Doom 3 BFG Edition); any other geometry works the same way.

### Quality presets

`quality="fast" | "normal" (default) | "high" | "max"`. For every block the encoder chooses a predictor
and an initial step size (`delta`), and the step size matters a lot, especially in small blocks: after a
few dozen samples the adapted step depends on where it started in a very irregular way, so
neighbouring start values can differ by several dB. The search therefore has two stages: a coarse
pass over all seven predictors with a few step sizes each picks the best predictor, then a dense
grid of step sizes (3 % apart) is tried for that one. The presets differ in how dense that grid is:
`fast` has no dense stage, `normal` tries every fourth grid point, `high` every point, and `max`
every point for the two best predictors.

How much that matters depends on the block size. With 128-sample blocks `normal` gains 5 to 8 dB
over `fast`, `high` another 2.5 to 5.5 dB, and `max` another 0.7 to 1.1 dB (tables below). With blocks of
500 to 1000 samples all presets from `normal` up are within 0.3 dB of each other, so `normal` is the
sensible choice there.

### Command line

```
python -m msadpcm encode in.wav -o out.wav --samples-per-block 128 --quality normal
python -m msadpcm encode *.wav -d out_dir --quality high -j 0          # all CPU cores
python -m msadpcm decode out.wav -o back.wav
python -m msadpcm info *.wav
```

`encode` shows a progress line on a terminal (nothing is printed when the output is redirected) and
reports the SNR of the result (it decodes the file again) unless `--no-verify` is given.

## Quality and speed

Measured on four real recordings (not included in the repository), all 22.05 kHz, 16-bit:
**speech** (53 s, mono), **sound effect** (a gunshot, 2.3 s, mono), **music** (168 s, mono) and
**speech with effects** (22 s, stereo; this file was itself decoded from MS ADPCM, so it is a second
generation). SNR in dB between the original and the decoded result, higher is better. Every file was
decoded with msadpcm's own decoder, which is bit-exact with FFmpeg's, so only the encoders differ.

**Default block size** (`256 * channels` bytes at 22 kHz: 512 bytes, which is 1012 samples per block
for mono and 500 for stereo), compared with FFmpeg 6.1's `adpcm_ms`:

| recording | ffmpeg | fast | normal | high | max |
|---|---|---|---|---|---|
| speech | 24.73 | 29.07 | 29.65 | 29.79 | 29.80 |
| sound effect | 31.03 | 32.22 | 32.78 | 32.79 | 32.89 |
| music | 22.23 | 25.34 | 25.90 | 26.04 | 26.06 |
| speech with effects (stereo) | 20.82 | 28.72 | 29.65 | 29.84 | 29.87 |

**Blocks of 128 samples** (70 bytes mono / 140 bytes stereo; FFmpeg only accepts power-of-two block
sizes, so it is not in this table):

| recording | fast | normal | high | max |
|---|---|---|---|---|
| speech | 32.13 | 37.69 | 40.18 | 41.28 |
| sound effect | 35.52 | 40.66 | 46.02 | 46.69 |
| music | 28.91 | 37.18 | 41.09 | 41.76 |
| speech with effects (stereo) | 31.43 | 38.94 | 41.89 | 42.80 |

(Version 0.1.2 reached 32.5 to 38.5 dB with `normal` and 36.5 to 41.3 dB with `max` on the same files;
see the [changelog](CHANGELOG.md).)

The gain of `normal` over FFmpeg's encoder at the default block size depends on the material: about
1.7 dB on the short sound effect and 8.8 dB on the stereo recording. On the synthetic signals of
`benchmarks/compare_ffmpeg.py` (anyone can run it) it ranges from about 2 to 9 dB. `msadpcm encode`
prints the SNR of every file it writes, so you can measure your own material.

**Several cores.** `workers=2` (or `-j 2`) took 5.3 s instead of 10 s for `max` on the 168 s music file,
2.2 s instead of 4.7 s for `high`, and 1.3 s instead of 2.0 s for `normal` with 512-byte blocks, on two
cores. Each worker process starts Python and NumPy, which costs a fraction of a second, so it only
pays for files that take several seconds to encode; smaller jobs are encoded in the calling process
automatically. On an i7-3770 with `-j 4`, a 6-minute mono file with 1024-byte blocks took 12.9 s instead of 32.3 s.

## How MS ADPCM blocks work

Each block starts with a header (`predictor, delta, sample1, sample2`) followed by 4-bit codes, high
nibble first. For stereo the header fields of both channels are interleaved by field, and every byte
after it holds one code of each channel (left in the high nibble). The decoder uses
`(s1*c1 + s2*c2) / 256` **truncated toward zero** and `delta = max(16, (table[nibble] * delta) >> 8)`,
exactly like FFmpeg and the Microsoft codec.

Details worth knowing:

* The first two samples of every block are stored verbatim, so each block starts exact.
* The last block is padded by repeating the final sample; the real length is written to the `fact`
  chunk and honoured by `read_wav` and `decode(n_samples=...)`.
* The header stores the initial delta as a signed 16-bit value; the encoder never writes more than 32767
  and avoids candidates whose delta would later exceed 32767, for decoders that keep it in 16 bits.
* Only 1 or 2 channels exist in this format.
* **Custom coefficient tables are not portable.** The format description allows extra coefficient pairs
  in the `fmt ` chunk, and msadpcm reads and writes them (`coeffs=`), but FFmpeg's decoder does not
  support them: it stops with an error on a predictor index of 7 or more, and silently decodes with the
  standard table when the first seven pairs differ, giving wrong audio. Use `coeffs=` only if you control
  the decoder as well. Microsoft's own codec is documented to accept extra pairs; that was not tested here.
* To embed blocks in another container, use `encode()` for the raw blocks and `format_chunk()` for the
  standard `fmt ` chunk body.

## API summary

| | |
|---|---|
| `encode(pcm, *, channels=None, block_align=None, samples_per_block=None, quality="normal", coeffs=None, progress=None, workers=1) -> bytes` | raw blocks |
| `decode(data, *, channels, block_align, n_samples=None, coeffs=None) -> int16[frames, channels]` | raw blocks |
| `write_wav(path, pcm, rate, *, adpcm=True, block_align=None, samples_per_block=None, quality="normal", coeffs=None, progress=None, workers=1)` | `adpcm=False` writes 16-bit PCM |
| `read_wav(path) -> (int16[frames, channels], rate)` | PCM 8/16/24/32, float, extensible, MS ADPCM |
| `wav_info(path) -> WavInfo`, `read_adpcm_raw(path)` | header / undecoded blocks |
| `format_chunk(rate, channels, block_align, coeffs=None)` | `fmt ` chunk body |
| `samples_per_block`, `block_align_for`, `default_block_align`, `snr_db` | helpers |

`progress` is `callback(done_blocks, total_blocks)`; encoding is vectorised over all blocks, so the
callback fires while a chunk of blocks is processed (every few dozen samples of it), not per block.
With `workers > 1` it fires when a chunk of blocks has been finished by a worker.

`workers` is the number of encoder processes (default 1, `0` = one per CPU). The result is identical for
any value. On Windows and macOS the workers are started by *spawn*, so a script that passes `workers`
greater than 1 must keep its entry point under `if __name__ == "__main__":`.

## Development

```
pip install -e .[test]
pytest
python benchmarks/compare_ffmpeg.py
```

The ffmpeg interoperability tests are skipped automatically when `ffmpeg` is not installed. The decoder is
also tested against an independent scalar re-implementation (`tests/helpers.py`). GitHub Actions runs the
tests on Windows, Linux and macOS with Python 3.9 to 3.13 (`.github/workflows/tests.yml`).

MIT licensed.
