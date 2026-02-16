import io
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf


def audio_bytes_to_mono16k_float32(audio_bytes: bytes, suffix: str = ".wav") -> Tuple[np.ndarray, int, str]:
    """
    Decode uploaded audio into mono 16kHz float32 waveform.
    Returns (waveform, sample_rate, normalized_wav_path).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src:
        src.write(audio_bytes)
        src_path = src.name

    wav_path = Path(tempfile.mkstemp(suffix=".wav")[1])
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            src_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        y, sr = sf.read(str(wav_path), always_2d=False)
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = np.mean(y, axis=1)
        y = np.asarray(y, dtype=np.float32)
        if y.size == 0:
            raise RuntimeError("Empty waveform after ffmpeg conversion")
        y = np.clip(y, -1.0, 1.0)
        return y, 16000, str(wav_path)
    except Exception:
        # Fallback decode path without ffmpeg
        y, sr = librosa.load(src_path, sr=16000, mono=True)
        y = np.asarray(y, dtype=np.float32)
        y = np.clip(y, -1.0, 1.0)
        sf.write(str(wav_path), y, 16000)
        return y, 16000, str(wav_path)


def bytes_to_audiofile(audio_bytes: bytes, suffix: str = ".wav") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        return tmp.name


def ensure_audio_path_16k_mono(path: str) -> str:
    wav_path = Path(tempfile.mkstemp(suffix=".wav")[1])
    cmd = ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000", str(wav_path)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(wav_path)
    except Exception:
        y, _sr = librosa.load(path, sr=16000, mono=True)
        sf.write(str(wav_path), y.astype(np.float32), 16000)
        return str(wav_path)

