import struct

import numpy as np
import pytest

import msadpcm
from msadpcm import wav

import helpers

# The 32 bytes of ADPCM extra data used by Doom 3 BFG .idwav files.
DOOM3_EXTRA = bytes.fromhex(
    "8000070000010000000200ff00000000c0004000f0000000cc0130ff880118ff"
)


def riff(*chunks):
    body = b"WAVE"
    for cid, data in chunks:
        body += struct.pack("<4sI", cid, len(data)) + data + (b"\0" if len(data) & 1 else b"")
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_fmt_chunk_is_standard_and_matches_doom3_extra_data():
    fmt = msadpcm.format_chunk(22050, 1, msadpcm.block_align_for(128, 1))
    assert len(fmt) == 50
    assert struct.unpack_from("<HHIIHHH", fmt) == (2, 1, 22050, 22050 * 70 // 128, 70, 4, 32)
    assert fmt[18:] == DOOM3_EXTRA
    fmt2 = msadpcm.format_chunk(44100, 2, 140)
    assert fmt2[18:] == DOOM3_EXTRA
    assert struct.unpack_from("<HH", fmt2, 12)[0] == 140


@pytest.mark.parametrize("channels", [1, 2])
def test_adpcm_wav_roundtrip(tmp_path, channels):
    pcm = helpers.make_stereo(7000)[:, :channels]
    p = tmp_path / "a.wav"
    msadpcm.write_wav(p, pcm, 22050, quality="fast")
    back, rate = msadpcm.read_wav(p)
    assert rate == 22050 and back.shape == pcm.shape       # fact chunk trims padding
    assert msadpcm.snr_db(pcm, back) > 25
    info = msadpcm.wav_info(p)
    assert info.is_adpcm and info.channels == channels and info.n_frames == 7000
    assert info.block_align == msadpcm.default_block_align(22050, channels)
    assert info.coeffs == msadpcm.STANDARD_COEFFS


def test_pcm_wav_roundtrip_is_lossless(tmp_path):
    pcm = helpers.make_stereo(1000)
    p = tmp_path / "p.wav"
    msadpcm.write_wav(p, pcm, 44100, adpcm=False)
    back, rate = msadpcm.read_wav(p)
    assert rate == 44100 and np.array_equal(back, pcm)
    assert msadpcm.wav_info(p).n_frames == 1000


def test_read_raw_blocks_and_explicit_geometry(tmp_path):
    p = tmp_path / "g.wav"
    msadpcm.write_wav(p, helpers.voice(1000), 22050, samples_per_block=128)
    raw, info = msadpcm.read_adpcm_raw(p)
    assert info.block_align == 70 and len(raw) == 70 * 8
    assert np.array_equal(
        msadpcm.decode(raw, channels=1, block_align=70, n_samples=1000)[:, 0],
        msadpcm.read_wav(p)[0][:, 0])


def test_custom_coeffs_are_stored_and_used(tmp_path):
    table = [(256, 0), (300, -40), (100, 100)]
    p = tmp_path / "c.wav"
    x = helpers.voice(3000)
    msadpcm.write_wav(p, x, 22050, coeffs=table, block_align=256)
    assert msadpcm.wav_info(p).coeffs == tuple(table)
    back, _ = msadpcm.read_wav(p)
    assert msadpcm.snr_db(x, back[:, 0]) > 25


def test_unknown_chunks_and_odd_padding_are_skipped(tmp_path):
    x = helpers.voice(500)
    pcm = x.astype("<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, 8000, 16000, 2, 16)
    p = tmp_path / "o.wav"
    p.write_bytes(riff((b"LIST", b"abc"), (b"fmt ", fmt), (b"junk", b"12345"), (b"data", pcm)))
    back, rate = msadpcm.read_wav(p)
    assert rate == 8000 and np.array_equal(back[:, 0], x)


def test_adpcm_without_fact_chunk_derives_length(tmp_path):
    x = helpers.voice(1000)
    data = msadpcm.encode(x, block_align=256)
    p = tmp_path / "nf.wav"
    p.write_bytes(riff((b"fmt ", msadpcm.format_chunk(22050, 1, 256)), (b"data", data)))
    info = msadpcm.wav_info(p)
    assert info.n_frames == 500 * 2
    assert len(msadpcm.read_wav(p)[0]) == 1000


def test_truncated_data_chunk_is_tolerated(tmp_path):
    data = msadpcm.encode(helpers.voice(1000), block_align=256)
    raw = riff((b"fmt ", msadpcm.format_chunk(22050, 1, 256)), (b"data", data))
    p = tmp_path / "t.wav"
    p.write_bytes(raw[:-100])           # header claims more bytes than the file holds
    pcm, _ = msadpcm.read_wav(p)
    assert len(pcm) > 500


@pytest.mark.parametrize("bits,dtype,scale", [(8, None, None), (24, None, None), (32, "<i4", 1 << 16)])
def test_other_pcm_widths_are_converted(tmp_path, bits, dtype, scale):
    x = helpers.voice(300)
    if bits == 8:
        body = ((x.astype(np.int32) >> 8) + 128).astype(np.uint8).tobytes()
        expect = ((x.astype(np.int32) >> 8) << 8)
    elif bits == 24:
        v = x.astype(np.int32) << 8
        body = b"".join(struct.pack("<i", int(s))[:3] for s in v)
        expect = x.astype(np.int32)
    else:
        body = (x.astype(np.int32) * scale).astype(dtype).tobytes()
        expect = x.astype(np.int32)
    fmt = struct.pack("<HHIIHH", 1, 1, 8000, 8000 * bits // 8, bits // 8, bits)
    p = tmp_path / "w.wav"
    p.write_bytes(riff((b"fmt ", fmt), (b"data", body)))
    back, _ = msadpcm.read_wav(p)
    assert np.array_equal(back[:, 0].astype(np.int32), expect)


def test_float_and_extensible_headers(tmp_path):
    x = helpers.voice(200)
    f = (x.astype(np.float32) / 32768).astype("<f4").tobytes()
    p = tmp_path / "f.wav"
    p.write_bytes(riff((b"fmt ", struct.pack("<HHIIHH", 3, 1, 8000, 32000, 4, 32)), (b"data", f)))
    assert np.array_equal(msadpcm.read_wav(p)[0][:, 0], x)

    # KSDATAFORMAT_SUBTYPE_PCM = 00000001-0000-0010-8000-00AA00389B71 (16 bytes)
    guid = struct.pack("<IHH8B", 1, 0, 0x10, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71)
    assert len(guid) == 16
    ext = struct.pack("<HHIIHH", 0xFFFE, 1, 8000, 16000, 2, 16) + struct.pack("<HHI", 22, 16, 4) + guid
    p2 = tmp_path / "e.wav"
    p2.write_bytes(riff((b"fmt ", ext), (b"data", x.astype("<i2").tobytes())))
    assert np.array_equal(msadpcm.read_wav(p2)[0][:, 0], x)


def test_bad_files(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"not a wav file at all")
    with pytest.raises(ValueError):
        msadpcm.read_wav(p)
    p.write_bytes(riff((b"fmt ", struct.pack("<HHIIHH", 1, 1, 8000, 16000, 2, 16))))
    with pytest.raises(ValueError, match="data"):
        msadpcm.read_wav(p)
    p.write_bytes(riff((b"data", b"\0\0")))
    with pytest.raises(ValueError, match="fmt"):
        msadpcm.read_wav(p)
    p.write_bytes(riff((b"fmt ", struct.pack("<HHIIHH", 0x55, 1, 8000, 16000, 2, 16)), (b"data", b"\0\0")))
    with pytest.raises(ValueError, match="unsupported"):
        msadpcm.read_wav(p)
    pcm = tmp_path / "pcm.wav"
    msadpcm.write_wav(pcm, helpers.voice(100), 8000, adpcm=False)
    with pytest.raises(ValueError, match="not an MS ADPCM"):
        msadpcm.read_adpcm_raw(pcm)
