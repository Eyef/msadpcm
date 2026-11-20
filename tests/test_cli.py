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
