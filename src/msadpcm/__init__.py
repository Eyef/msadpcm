"""msadpcm - Microsoft ADPCM encoder/decoder in pure Python + NumPy."""
from .codec import (
    ADAPTATION_TABLE,
    QUALITY_LEVELS,
    STANDARD_COEFFS,
    block_align_for,
    decode,
    default_block_align,
    encode,
    samples_per_block,
    snr_db,
)
from .wav import WavInfo, format_chunk, read_adpcm_raw, read_wav, wav_info, write_wav

__version__ = "0.1.3"

__all__ = [
    "ADAPTATION_TABLE",
    "QUALITY_LEVELS",
    "STANDARD_COEFFS",
    "WavInfo",
    "block_align_for",
    "decode",
    "default_block_align",
    "encode",
    "format_chunk",
    "read_adpcm_raw",
    "read_wav",
    "samples_per_block",
    "snr_db",
    "wav_info",
    "write_wav",
]
