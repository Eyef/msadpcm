# Changelog

## 0.2.0

**Encoder quality.** The initial step size (`delta`) search was rewritten. With small blocks
(for example 128 samples, the 70-byte mono / 140-byte stereo blocks used by some game engines) the
result depends strongly and irregularly on that value, and the old grid of a few dozen candidates
per predictor missed much of what a fine search finds. The new search first picks the predictor
with a coarse pass (all seven predictors, a few step sizes each), then tries a dense grid of step
sizes (3 % apart) for the best predictor only.

Measured on four real recordings at 22.05 kHz (speech, a gunshot, music, stereo speech with
effects), 128 samples per block, SNR in dB:

| preset | 0.1.2 | 0.2.0 |
|---|---|---|
| `normal` | 32.5 to 38.5 | 37.2 to 40.7 |
| `high`   | 34.0 to 39.5 | 40.2 to 46.0 |
| `max`    | 36.5 to 41.3 | 41.3 to 46.7 |

With 512 to 1024-byte blocks the differences between presets were already small and stay small
(a gain of 0.2 to 0.4 dB for `normal`; `max` is 0.06 dB lower on one file and 0.15 to 0.25 dB higher on the others). `fast` is unchanged. Speed is about the same as before for `normal`
and `max` (`max` is often faster), and `high` is up to 1.4 times slower because it now searches
more candidates. The meaning of the presets changed: `high` is now roughly what `max` used to be
in cost, with better results.

**New:** `workers=N` (`encode`, `write_wav`, `convert_file`) and `-j N` / `--workers N` on the command
line encode with several processes (`0` = one per CPU). The output is byte-for-byte identical to
the single-process result. About 1.8 to 2 times faster on two cores for files of a few minutes; for
short inputs the process start-up costs more than it saves, so jobs of less than about a second of
single-process work are encoded in the calling process whatever `workers` says. On a 4-core
i7-3770 a 6-minute mono file took 12.9 s with `-j 4` instead of 32.3 s. On Windows and macOS a script that
uses `workers` must be guarded by `if __name__ == "__main__":`.

**Other:**
* The README's description of the default block size was wrong for stereo: 22 kHz stereo gets 512-byte
  blocks (500 samples), not 1024 bytes; the tables were measured with the correct size.
* Work is processed in smaller chunks that fit the CPU cache: roughly 10 to 20 % faster on the test machine.
* `QUALITY_LEVELS` has new keys (`coarse`, `keep`, `step`, `window`); the old `mults` and `refine`
  are gone. Only code that read or replaced that dictionary is affected.
* GitHub Actions: tests on Windows, Linux and macOS with Python 3.9 to 3.13, and a publishing
  workflow for PyPI (Trusted Publishing).

## 0.1.2

* `wav_info` no longer trusts an oversized `fact` chunk (some tools write twice the real length
  for stereo); the README was corrected and extended.

## 0.1.1

* Progress line on a terminal, default quality changed to `normal`.

## 0.1.0

* First release: MS ADPCM encoder and decoder, WAV reader/writer, command line tool.
