from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _window_ranges(n_samples: int, sr: int, win_sec: float = 2.0, hop_sec: float = 1.0) -> List[Tuple[int, int]]:
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    if n_samples <= win:
        return [(0, n_samples)]
    spans = []
    i = 0
    while i + win <= n_samples:
        spans.append((i, i + win))
        i += hop
    if spans and spans[-1][1] < n_samples:
        spans.append((n_samples - win, n_samples))
    return spans


def _norm01(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo = float(np.percentile(scores, 5))
    hi = float(np.percentile(scores, 95))
    if hi - lo < 1e-8:
        return np.clip(scores, 0.0, 1.0)
    return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)


def _mel_vec(segment: np.ndarray, sr: int, n_mels: int = 64) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return np.concatenate([mel_db.mean(axis=1), mel_db.std(axis=1)]).astype(np.float32)


class _AE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hid = max(16, dim // 2)
        code = max(8, dim // 4)
        self.enc = nn.Sequential(nn.Linear(dim, hid), nn.ReLU(), nn.Linear(hid, code), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(code, hid), nn.ReLU(), nn.Linear(hid, dim))

    def forward(self, x):
        return self.dec(self.enc(x))


def compute_acoustic_anomaly(
    waveform: np.ndarray,
    sr: int,
    baseline_waveform: np.ndarray | None = None,
    baseline_seconds: int = 25,
    epochs: int = 8,
) -> Dict[str, object]:
    spans = _window_ranges(len(waveform), sr, 2.0, 1.0)
    segs = [waveform[s:e] for s, e in spans]
    if not segs:
        return {"timeline": [], "scores": np.array([], dtype=np.float32), "events": []}

    feats = np.stack([_mel_vec(seg, sr) for seg in segs], axis=0)

    if baseline_waveform is not None and baseline_waveform.size > 0:
        b_spans = _window_ranges(len(baseline_waveform), sr, 2.0, 1.0)
        b_segs = [baseline_waveform[s:e] for s, e in b_spans]
    else:
        b_len = min(len(waveform), int(baseline_seconds * sr))
        b_spans = _window_ranges(b_len, sr, 2.0, 1.0)
        b_segs = [waveform[s:e] for s, e in b_spans]
    b_feats = np.stack([_mel_vec(seg, sr) for seg in b_segs], axis=0)

    x_train = torch.tensor(b_feats, dtype=torch.float32)
    x_full = torch.tensor(feats, dtype=torch.float32)

    model = _AE(feats.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(max(3, epochs)):
        optimizer.zero_grad()
        rec = model(x_train)
        loss = criterion(rec, x_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        rec = model(x_full).cpu().numpy()
    err = np.mean((feats - rec) ** 2, axis=1)
    scores = _norm01(err.astype(np.float32))

    timeline = [{"t": float(i), "anomaly_score": float(s)} for i, s in enumerate(scores)]
    events = []
    for i, s in enumerate(scores):
        if s > 0.75:
            events.append(
                {
                    "start": float(i),
                    "end": float(i + 2.0),
                    "event_type": "acoustic_anomaly",
                    "score": float(s),
                    "label": "Acoustic Pattern Shift",
                    "details": "Mel-spectrum deviation exceeded threshold (0.75).",
                }
            )

    return {"timeline": timeline, "scores": scores, "events": events}

