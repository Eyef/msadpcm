# msadpcm

Microsoft ADPCM (`WAVE_FORMAT_ADPCM`, tag `0x0002`) **encoder and decoder** in pure
Python + NumPy. No external programs and no compiled extensions.

* any block size (`block_align` in bytes, or `samples_per_block`), mono and stereo
* decoder output is **bit-exact with FFmpeg's `adpcm_ms`** (checked in the test suite, both directions)
* encoder searches all 7 standard predictors and a grid of initial step sizes for every block, with
  the same arithmetic as the decoder, so the quality it measures is the quality you get
* reads and writes complete WAV files (`fact` chunk included), or works on raw blocks in memory
* optional custom coefficient tables
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

`quality="fast" | "normal" (default) | "high" | "max"`: how many candidate step sizes are tried per
block (all presets try all seven predictors). The slower presets run a finer search.

How much that matters depends on the block size. With blocks of 512 to 1024 bytes `normal`, `high` and
`max` are within 0.4 dB of each other (`fast` is up to 0.9 dB below `max`). With very small blocks (for
example 128 samples) the initial step size has far more influence: `high` gains about 1 to 1.5 dB over
`normal` and `max` 3 to 4 dB (see the tables below), so prefer them there.

### Command line

```
python -m msadpcm encode in.wav -o out.wav --samples-per-block 128 --quality normal
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

**Blocks of 512 bytes (mono) / 1024 bytes (stereo)**, 1012 samples per block, compared with FFmpeg 6.1's
`adpcm_ms` (SNR, dB):


| recording | ffmpeg | fast | normal | high | max |
|---|---|---|---|---|---|
| speech | 24.73 | 29.07 | 29.43 | 29.52 | 29.65 |
| sound effect | 31.03 | 32.22 | 32.62 | 32.78 | 32.95 |
| music | 22.23 | 25.34 | 25.68 | 25.75 | 25.90 |
| speech with effects (stereo) | 20.82 | 28.72 | 29.25 | 29.37 | 29.64 |

**Blocks of 128 samples** (70 bytes mono / 140 bytes stereo; FFmpeg only accepts power-of-two block
sizes, so it is not in this table):

| recording | fast | normal | high | max |
|---|---|---|---|---|
| speech | 32.13 | 35.05 | 36.20 | 38.24 |
| sound effect | 35.52 | 38.49 | 39.49 | 41.29 |
| music | 28.91 | 32.53 | 33.96 | 36.45 |
| speech with effects (stereo) | 31.43 | 34.66 | 36.07 | 38.48 |

The gain of `normal` over FFmpeg's encoder depends on the material: about 1.6 dB on the short sound
effect and 8 dB on the stereo recording. On the synthetic signals of `benchmarks/compare_ffmpeg.py`
(anyone can run it) it ranges from about 2 to 9 dB. `msadpcm encode` prints the SNR of every file it
writes, so you can measure your own material.


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
* To embed blocks in another container, use `encode()` for the raw blocks and `format_chunk()` for the
  standard `fmt ` chunk body.

## API summary

| | |
|---|---|
| `encode(pcm, *, channels=None, block_align=None, samples_per_block=None, quality="normal", coeffs=None, progress=None) -> bytes` | raw blocks |
| `decode(data, *, channels, block_align, n_samples=None, coeffs=None) -> int16[frames, channels]` | raw blocks |
| `write_wav(path, pcm, rate, *, adpcm=True, block_align=None, samples_per_block=None, quality="normal", coeffs=None, progress=None)` | `adpcm=False` writes 16-bit PCM |
| `read_wav(path) -> (int16[frames, channels], rate)` | PCM 8/16/24/32, float, extensible, MS ADPCM |
| `wav_info(path) -> WavInfo`, `read_adpcm_raw(path)` | header / undecoded blocks |
| `format_chunk(rate, channels, block_align, coeffs=None)` | `fmt ` chunk body |
| `samples_per_block`, `block_align_for`, `default_block_align`, `snr_db` | helpers |

`progress` is `callback(done_blocks, total_blocks)`; encoding is vectorised over all blocks, so the
callback fires once per processed chunk of blocks (a few hundred blocks), not per block.

## Development

```
pip install -e .[test]
pytest
python benchmarks/compare_ffmpeg.py
```

The ffmpeg interoperability tests are skipped automatically when `ffmpeg` is not installed. The decoder is
also tested against an independent scalar re-implementation (`tests/helpers.py`).

MIT licensed.
