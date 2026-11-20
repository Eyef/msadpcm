# msadpcm

Microsoft ADPCM (`WAVE_FORMAT_ADPCM`, tag `0x0002`) **encoder and decoder** in pure
Python + NumPy. No FFmpeg, no compiled extensions, no multiprocessing.

* any block size (`block_align` in bytes, or `samples_per_block`), mono and stereo
* decoder output is **bit-exact with FFmpeg's `adpcm_ms`** (checked in the test suite, both directions)
* encoder searches all 7 standard predictors and a grid of initial step sizes for every block, with
  the same arithmetic as the decoder, so the quality it measures is the quality you get
* reads and writes complete WAV files (`fact` chunk included), or works on raw blocks in memory
* optional custom coefficient tables
* command line tool and a small tkinter demo (`examples/gui_demo.py`)

```
pip install msadpcm            # once published; until then: pip install .
```

## Quick start

```python
import numpy as np
import msadpcm

# whole files
msadpcm.write_wav("out.wav", pcm, 22050)                 # pcm: int16 (frames,) or (frames, 2)
pcm, rate = msadpcm.read_wav("out.wav")                  # ADPCM or PCM in, int16 (frames, channels) out

# raw blocks
blocks = msadpcm.encode(pcm, block_align=512, quality="high")
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

### Quality presets

`quality="fast" | "normal" | "high" (default) | "max"`: how many candidate step sizes are tried per
block. All presets try all seven predictors.

### Command line

```
python -m msadpcm encode in.wav -o out.wav --samples-per-block 128 --quality high
python -m msadpcm decode out.wav -o back.wav
python -m msadpcm info *.wav
```

`encode` prints the SNR of the result (it decodes the file again) unless `--no-verify` is given.

## Quality and speed

SNR in dB (higher is better), 6 s of mono audio at 22.05 kHz, run with
`python benchmarks/compare_ffmpeg.py` (ffmpeg 6.1). All files were decoded with msadpcm's decoder,
which is bit-exact with FFmpeg's, so the differences come from the encoders only.

| signal | block | ffmpeg `adpcm_ms` | fast | high | max |
|---|---|---|---|---|---|
| voice-like | 256 | 30.40 | 39.20 | 39.88 | 40.04 |
| music-like | 256 | 34.54 | 42.14 | 42.59 | 42.76 |
| white noise | 256 | 14.12 | 16.59 | 16.72 | 16.77 |
| voice-like | 1024 | 30.87 | 38.60 | 38.86 | 38.97 |
| music-like | 1024 | 34.54 | 39.19 | 39.38 | 39.46 |
| white noise | 1024 | 14.13 | 16.52 | 16.57 | 16.59 |

Caveats: the signals are synthetic (the test suite needs no audio assets), and harmonic signals
reward good predictor choice more than typical recordings do, so expect smaller gains on real
material. Run the script on your own files before relying on these numbers.

Encoding time is roughly 0.1x to 0.4x of real time at `high` on one core (about 10-20x faster than
real time for typical block sizes). Memory is bounded regardless of block size.

## Format notes

Blocks are `predictor, delta, sample1, sample2` followed by 4-bit codes (high nibble first); stereo
stores both channels' header fields interleaved by field and both channels' codes in one byte (left in the
high nibble). The decoder uses `(s1*c1 + s2*c2) / 256` **truncated toward zero** and
`delta = max(16, (table[nibble] * delta) >> 8)`, exactly like FFmpeg and the Microsoft codec.

Details worth knowing:

* The first two samples of every block are stored verbatim, so each block starts exact.
* The last block is padded by repeating the final sample; the real length is written to the `fact`
  chunk and honoured by `read_wav`/`decode(n_samples=...)`.
* The header stores the initial delta as a signed 16-bit value; the encoder never writes more than 32767
  and avoids candidates whose delta would later exceed 32767, for decoders that keep it in 16 bits.
* Only 1 or 2 channels exist in this format.

### Game audio with fixed block sizes

Some games expect a particular block geometry. `samples_per_block=128` produces the 70/140-byte blocks
and the 32-byte extra data (`wSamplesPerBlock=128, wNumCoef=7` + the standard coefficients) that Doom 3 BFG
Edition uses, which is the standard layout, nothing proprietary:

```python
fmt = msadpcm.format_chunk(22050, 1, msadpcm.block_align_for(128, 1))   # 50 bytes, extra data included
raw = msadpcm.encode(pcm, samples_per_block=128)                         # blocks without any container
```

## API summary

| | |
|---|---|
| `encode(pcm, *, channels=None, block_align=None, samples_per_block=None, quality="high", coeffs=None, progress=None) -> bytes` | raw blocks |
| `decode(data, *, channels, block_align, n_samples=None, coeffs=None) -> int16[frames, channels]` | raw blocks |
| `write_wav(path, pcm, rate, *, adpcm=True, block_align=None, samples_per_block=None, quality="high", coeffs=None, progress=None)` | `adpcm=False` writes 16-bit PCM |
| `read_wav(path) -> (int16[frames, channels], rate)` | PCM 8/16/24/32, float, extensible, MS ADPCM |
| `wav_info(path) -> WavInfo`, `read_adpcm_raw(path)` | header / undecoded blocks |
| `format_chunk(rate, channels, block_align, coeffs=None)` | `fmt ` chunk body |
| `samples_per_block`, `block_align_for`, `default_block_align`, `snr_db` | helpers |

`progress` is `callback(done_blocks, total_blocks)`; encoding is vectorised over all blocks, so the
callback fires per processed chunk of blocks, not per block.

## Development

```
pip install -e .[test]
pytest
python benchmarks/compare_ffmpeg.py
```

The ffmpeg interoperability tests are skipped automatically when `ffmpeg` is not installed. The decoder is
also tested against an independent scalar re-implementation (`tests/helpers.py`).

MIT licensed.
