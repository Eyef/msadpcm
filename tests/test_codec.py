import numpy as np
import pytest

import msadpcm
from msadpcm import codec

import helpers
from helpers import reference_decode


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def test_standard_block_sizes():
    assert msadpcm.samples_per_block(256, 1) == 500
    assert msadpcm.samples_per_block(512, 2) == 500
    assert msadpcm.samples_per_block(70, 1) == 128
    assert msadpcm.samples_per_block(140, 2) == 128


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("spb", [4, 12, 128, 500, 2040])
def test_block_align_roundtrip(channels, spb):
    ba = msadpcm.block_align_for(spb, channels)
    assert msadpcm.samples_per_block(ba, channels) == spb


def test_geometry_errors():
    with pytest.raises(ValueError):
        msadpcm.block_align_for(127, 1)        # odd mono block
    with pytest.raises(ValueError):
        msadpcm.samples_per_block(7, 1)        # no room for a nibble
    with pytest.raises(ValueError):
        msadpcm.samples_per_block(100, 3)      # unsupported channel count
    with pytest.raises(ValueError):
        msadpcm.encode(np.zeros(10, np.int16), block_align=70, samples_per_block=128)


def test_default_block_align_matches_windows_acm():
    assert msadpcm.default_block_align(11025, 1) == 256
    assert msadpcm.default_block_align(22050, 1) == 512
    assert msadpcm.default_block_align(44100, 1) == 1024
    assert msadpcm.default_block_align(44100, 2) == 2048


# --------------------------------------------------------------------------
# decoder == independent scalar reference, bit for bit
# --------------------------------------------------------------------------
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("quality", list(codec.QUALITY_LEVELS))
def test_decoder_matches_reference(channels, quality):
    pcm = helpers.make_stereo(6000)[:, :channels]
    ba = 70 * channels
    data = msadpcm.encode(pcm, block_align=ba, quality=quality)
    got = msadpcm.decode(data, channels=channels, block_align=ba)
    assert np.array_equal(got, reference_decode(data, channels, ba))


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("ba", [20, 70, 256, 1024, 2048])
def test_block_sizes(channels, ba):
    pcm = helpers.make_stereo(5000)[:, :channels]
    data = msadpcm.encode(pcm, block_align=ba * channels, quality="fast")
    got = msadpcm.decode(data, channels=channels, block_align=ba * channels)
    assert np.array_equal(got, reference_decode(data, channels, ba * channels))
    assert len(data) % (ba * channels) == 0


@pytest.mark.parametrize("gen", [helpers.full_scale_noise, helpers.square, helpers.impulses])
def test_hostile_signals_stay_bit_exact(gen):
    """Clipping, huge deltas and sudden jumps must not break encoder/decoder agreement."""
    x = gen(4000)
    data = msadpcm.encode(x, block_align=256, quality="normal")
    got = msadpcm.decode(data, channels=1, block_align=256)
    assert np.array_equal(got, reference_decode(data, 1, 256))


def test_custom_coefficient_table():
    table = [(256, 0), (400, -100), (100, 50)]
    x = helpers.voice(3000)
    data = msadpcm.encode(x, block_align=128, coeffs=table, quality="normal")
    got = msadpcm.decode(data, channels=1, block_align=128, coeffs=table)
    assert np.array_equal(got, reference_decode(data, 1, 128, coeffs=table))
    assert msadpcm.snr_db(x, got[:len(x), 0]) > 25
    assert max(data[i] for i in range(0, len(data), 128)) < len(table)


def test_invalid_predictor_index_is_rejected():
    data = bytearray(msadpcm.encode(helpers.voice(500), block_align=256))
    data[0] = 7  # standard table has predictors 0..6
    with pytest.raises(ValueError, match="predictor"):
        msadpcm.decode(bytes(data), channels=1, block_align=256)


# --------------------------------------------------------------------------
# encoder behaviour
# --------------------------------------------------------------------------
def test_encoder_error_estimate_equals_real_error():
    """The encoder's internal simulation must equal what a decoder really produces."""
    x = helpers.music(8 * 128).reshape(8, 128)
    preds, deltas, nibs, errs = codec._encode_rows(x, None, "high", None)
    data = codec._pack_blocks(x.astype(np.int32), preds, deltas, nibs, 1, 70)
    dec = msadpcm.decode(data, channels=1, block_align=70)[:, 0].reshape(8, 128)
    real = ((x.astype(np.float64) - dec) ** 2).sum(axis=1)
    assert np.array_equal(real, errs)


def test_first_two_samples_of_each_block_are_exact():
    x = helpers.voice(128 * 10)
    data = msadpcm.encode(x, samples_per_block=128)
    dec = msadpcm.decode(data, channels=1, block_align=70)[:, 0]
    for b in range(10):
        assert np.array_equal(dec[b * 128:b * 128 + 2], x[b * 128:b * 128 + 2])


def test_initial_delta_is_valid_header_value():
    x = helpers.full_scale_noise(128 * 20)
    data = np.frombuffer(msadpcm.encode(x, samples_per_block=128), np.uint8).reshape(-1, 70)
    deltas = data[:, 1].astype(int) | (data[:, 2].astype(int) << 8)
    assert deltas.min() >= 16 and deltas.max() <= 32767


def test_quality_levels_are_ordered():
    x = helpers.music(22050 * 2)
    snr = {}
    for q in codec.QUALITY_LEVELS:
        d = msadpcm.decode(msadpcm.encode(x, block_align=256, quality=q), channels=1, block_align=256)
        snr[q] = msadpcm.snr_db(x, d[:len(x), 0])
    # Refinement is centred on each preset's own winner, so neighbouring presets
    # may differ by a few hundredths of a dB the "wrong" way; the trend must hold.
    assert snr["fast"] <= snr["normal"] + 0.25
    assert snr["normal"] <= snr["high"] + 0.25
    assert snr["high"] <= snr["max"] + 0.25
    assert snr["max"] > snr["fast"] - 0.01
    assert snr["fast"] > 30


def test_unknown_quality_rejected():
    with pytest.raises(ValueError):
        msadpcm.encode(helpers.voice(500), quality="ultra")


def test_progress_callback():
    calls = []
    msadpcm.encode(helpers.voice(128 * 50), samples_per_block=128,
                   progress=lambda d, t: calls.append((d, t)))
    assert calls[0][0] == 0 and calls[-1][0] == calls[-1][1] == 50
    assert [c[0] for c in calls] == sorted(c[0] for c in calls)


def test_many_chunks_give_same_result(monkeypatch):
    """Chunked processing must not change the output."""
    x = helpers.voice(128 * 40)
    whole = msadpcm.encode(x, samples_per_block=128, quality="normal")
    monkeypatch.setattr(codec, "_LANES_PER_CHUNK", 64)
    assert msadpcm.encode(x, samples_per_block=128, quality="normal") == whole


# --------------------------------------------------------------------------
# inputs / edge cases
# --------------------------------------------------------------------------
def test_empty_input():
    assert msadpcm.encode(np.zeros(0, np.int16)) == b""
    out = msadpcm.decode(b"", channels=2, block_align=512)
    assert out.shape == (0, 2)


@pytest.mark.parametrize("n", [1, 2, 3, 127, 128, 129])
def test_short_inputs_and_padding(n):
    x = helpers.voice(n)
    data = msadpcm.encode(x, samples_per_block=128)
    assert len(data) == 70 * ((n + 127) // 128)
    dec = msadpcm.decode(data, channels=1, block_align=70, n_samples=n)
    assert dec.shape == (n, 1)
    # the tail is padded by repeating the last sample (not with zeros -> no click)
    pad = (-n) % 128
    manual = np.concatenate([x, np.full(pad, x[-1], dtype=np.int16)])
    assert data == msadpcm.encode(manual, samples_per_block=128)


def test_silence_stays_silent():
    data = msadpcm.encode(np.zeros(1000, np.int16), block_align=256)
    assert not msadpcm.decode(data, channels=1, block_align=256).any()


def test_input_forms_are_equivalent():
    x = helpers.voice(2000)
    ref = msadpcm.encode(x, block_align=256)
    assert msadpcm.encode(x.reshape(-1, 1), block_align=256) == ref
    assert msadpcm.encode(x.astype("<i2").tobytes(), block_align=256) == ref
    assert msadpcm.encode(list(map(int, x)), block_align=256) == ref
    st = np.stack([x, x[::-1]], axis=1)
    assert msadpcm.encode(st.reshape(-1), channels=2, block_align=512) == msadpcm.encode(st, block_align=512)


def test_float_input_matches_int16():
    x = helpers.voice(2000)
    assert msadpcm.encode(x.astype(np.float64) / 32768, block_align=256) == msadpcm.encode(x, block_align=256)


def test_rejects_wrong_integer_widths_and_shapes():
    with pytest.raises(TypeError):
        msadpcm.encode(np.zeros(100, np.int32))
    with pytest.raises(ValueError):
        msadpcm.encode([0, 1, 40000])          # list values must fit in int16
    with pytest.raises(TypeError):
        msadpcm.encode(np.zeros(100, np.uint8))
    with pytest.raises(ValueError):
        msadpcm.encode(np.zeros((10, 3), np.int16))
    with pytest.raises(ValueError):
        msadpcm.encode(np.zeros(9, np.int16), channels=2)
    with pytest.raises(ValueError):
        msadpcm.encode(np.zeros((10, 2), np.int16), channels=1)


def test_partial_trailing_block_is_decoded():
    x = helpers.voice(128 * 3)
    data = msadpcm.encode(x, samples_per_block=128)
    cut = data[: 70 * 2 + 7 + 10]       # two full blocks + header + 10 nibble bytes
    dec = msadpcm.decode(cut, channels=1, block_align=70)
    assert len(dec) == 128 * 2 + 2 + 20
    assert np.array_equal(dec[:256], msadpcm.decode(data, channels=1, block_align=70)[:256])
    assert np.array_equal(dec, msadpcm.decode(data, channels=1, block_align=70)[:len(dec)])


def test_snr_helper():
    a = np.array([1000, -1000, 500], np.int16)
    assert msadpcm.snr_db(a, a) == float("inf")
    assert 15 < msadpcm.snr_db(a, a + 100) < 25
