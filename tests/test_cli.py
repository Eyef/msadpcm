import importlib.util
from pathlib import Path

import numpy as np
import pytest

import msadpcm
from msadpcm import cli

import helpers


@pytest.fixture
def pcm_file(tmp_path):
    p = tmp_path / "in.wav"
    msadpcm.write_wav(p, helpers.make_stereo(6000), 22050, adpcm=False)
    return p


def test_convert_file_encode_then_decode(tmp_path, pcm_file):
    seen = []
    enc = cli.convert_file(pcm_file, tmp_path / "e.wav", samples_per_block=128,
                           quality="fast", progress=lambda d, t: seen.append(d))
    assert enc.mode == "encode" and enc.block_align == 140 and enc.samples_per_block == 128
    assert enc.snr_db > 25 and enc.ratio > 2 and seen
    dec = cli.convert_file(enc.dst, tmp_path / "d.wav", mode="decode")
    assert dec.frames == 6000 and dec.channels == 2
    assert msadpcm.snr_db(msadpcm.read_wav(pcm_file)[0], msadpcm.read_wav(dec.dst)[0]) == enc.snr_db


def test_default_output_names(pcm_file):
    enc = cli.convert_file(pcm_file, quality="fast")
    assert enc.dst.name == "in.adpcm.wav" and enc.dst.exists()
    assert cli.convert_file(enc.dst, mode="decode").dst.name == "in.adpcm.pcm.wav"


def test_cli_main(tmp_path, pcm_file, capsys):
    out = tmp_path / "out"
    assert cli.main(["encode", str(pcm_file), "-d", str(out), "--quality", "fast",
                     "--samples-per-block", "128"]) == 0
    produced = out / "in.adpcm.wav"
    assert produced.exists()
    assert cli.main(["info", str(produced)]) == 0
    text = capsys.readouterr().out
    assert "MS ADPCM" in text and "block 140 bytes (128 samples)" in text and "6000 frames" in text
    assert cli.main(["decode", str(produced), "-o", str(tmp_path / "back.wav")]) == 0
    assert msadpcm.read_wav(tmp_path / "back.wav")[0].shape == (6000, 2)


def test_cli_reports_errors_and_continues(tmp_path, pcm_file, capsys):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"garbage")
    assert cli.main(["encode", str(bad), str(pcm_file), "--quality", "fast"]) == 1
    assert (tmp_path / "in.adpcm.wav").exists()
    assert "bad.wav" in capsys.readouterr().err


def test_gui_demo_block_choices():
    """The demo module must import without tkinter; only main() needs it."""
    path = Path(__file__).resolve().parents[1] / "examples" / "gui_demo.py"
    spec = importlib.util.spec_from_file_location("gui_demo", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.parse_block_choice("128 samples (70/140-byte blocks)") == (None, 128)
    assert mod.parse_block_choice("1024 bytes") == (1024, None)
    for label in mod.BLOCK_CHOICES:                  # every choice is accepted by the encoder
        ba, spb = mod.parse_block_choice(label)
        msadpcm.encode(helpers.voice(300), block_align=ba, samples_per_block=spb, quality="fast")


# --------------------------------------------------------------------------
# progress line and defaults
# --------------------------------------------------------------------------
class FakeStream:
    def __init__(self, tty):
        self.tty, self.chunks = tty, []

    def isatty(self):
        return self.tty

    def write(self, text):
        self.chunks.append(text)

    def flush(self):
        pass

    @property
    def text(self):
        return "".join(self.chunks)


def test_progress_line_on_terminal():
    s = FakeStream(tty=True)
    line = cli._ProgressLine("a.wav", s)
    for done in range(0, 101, 5):
        line(done, 100)
    line(100, 100)                      # repeated percentage is not redrawn twice
    assert "a.wav: encoding 0%" in s.text and "a.wav: encoding 100%" in s.text
    assert s.text.count("encoding 50%") == 1
    line.clear()
    assert s.text.endswith("\r") and s.text.count("\r") > 20


def test_progress_line_silent_when_redirected():
    s = FakeStream(tty=False)
    line = cli._ProgressLine("a.wav", s)
    line(0, 10)
    line(10, 10)
    line.clear()
    assert s.text == ""


def test_cli_shows_progress_only_on_a_terminal(tmp_path, pcm_file, monkeypatch, capsys):
    out = tmp_path / "o.wav"
    assert cli.main(["encode", str(pcm_file), "-o", str(out), "--quality", "fast"]) == 0
    assert capsys.readouterr().err == ""              # pytest's stderr is not a terminal

    fake = FakeStream(tty=True)
    monkeypatch.setattr(cli.sys, "stderr", fake)
    assert cli.main(["encode", str(pcm_file), "-o", str(out), "--quality", "fast"]) == 0
    assert "in.wav: encoding 100%" in fake.text
    assert fake.text.endswith("\r")                   # line wiped before the result is printed


def test_progress_callback_is_smooth(monkeypatch):
    """Many small steps, so a console shows more than a jump from 0% to 100%."""
    seen = []
    msadpcm.encode(helpers.voice(500 * 3000), block_align=256, quality="normal",
                   progress=lambda d, t: seen.append(d))
    assert len(seen) >= 8 and seen[0] == 0 and seen[-1] == 3000
    assert all(b > a for a, b in zip(seen, seen[1:]))      # strictly increasing, never repeats


def test_progress_is_smooth_within_a_single_chunk():
    """Progress comes from inside the encoder loop, not only between chunks of blocks."""
    seen = []
    msadpcm.encode(helpers.voice(500 * 200), block_align=256, quality="normal",
                   progress=lambda d, t: seen.append(d))      # 500 samples/block: 200 blocks, one chunk
    assert len(seen) >= 20 and seen[-1] == 200


def test_default_quality_is_normal():
    x = helpers.voice(3000)
    default = msadpcm.encode(x, block_align=256)
    assert default == msadpcm.encode(x, block_align=256, quality="normal")
    assert default != msadpcm.encode(x, block_align=256, quality="fast")
    assert cli._build_parser().parse_args(["encode", "x.wav"]).quality == "normal"
