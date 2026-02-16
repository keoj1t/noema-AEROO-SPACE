from typing import Dict, List, Tuple

import librosa
import numpy as np


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


def _embedding_speechbrain(segment: np.ndarray, sr: int) -> np.ndarray:
    try:
        import torch
        from speechbrain.pretrained import EncoderClassifier

        if not hasattr(_embedding_speechbrain, "_sb_model"):
            _embedding_speechbrain._sb_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"},
            )
        model = _embedding_speechbrain._sb_model
        wav = torch.from_numpy(segment.astype(np.float32)).unsqueeze(0)
        emb = model.encode_batch(wav).detach().cpu().numpy().reshape(-1)
        return emb.astype(np.float32)
    except Exception:
        # MFCC fallback embedding
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)
        return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)


def _norm01(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo = float(np.percentile(scores, 5))
    hi = float(np.percentile(scores, 95))
    if hi - lo < 1e-8:
        return np.clip(scores, 0.0, 1.0)
    return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)


def compute_stress_timeline(
    waveform: np.ndarray,
    sr: int,
    baseline_waveform: np.ndarray | None = None,
    baseline_seconds: int = 25,
) -> Dict[str, object]:
    spans = _window_ranges(len(waveform), sr, 2.0, 1.0)
    segments = [waveform[s:e] for s, e in spans]
    if not segments:
        return {"timeline": [], "scores": np.array([], dtype=np.float32)}

    if baseline_waveform is not None and baseline_waveform.size > 0:
        b_spans = _window_ranges(len(baseline_waveform), sr, 2.0, 1.0)
        b_segments = [baseline_waveform[s:e] for s, e in b_spans]
    else:
        b_len = min(len(waveform), int(baseline_seconds * sr))
        b_spans = _window_ranges(b_len, sr, 2.0, 1.0)
        b_segments = [waveform[s:e] for s, e in b_spans]

    emb_base = np.stack([_embedding_speechbrain(seg, sr) for seg in b_segments], axis=0)
    ref = emb_base.mean(axis=0)
    ref_norm = np.linalg.norm(ref) + 1e-8

    dev = []
    points = []
    for idx, seg in enumerate(segments):
        emb = _embedding_speechbrain(seg, sr)
        sim = float(np.dot(emb, ref) / ((np.linalg.norm(emb) + 1e-8) * ref_norm))
        score = 1.0 - (sim + 1.0) / 2.0
        dev.append(score)
        t0 = idx * 1.0
        points.append({"t": t0, "stress_score": score})

    dev_arr = _norm01(np.asarray(dev, dtype=np.float32))
    for i, p in enumerate(points):
        p["stress_score"] = float(dev_arr[i])
    return {"timeline": points, "scores": dev_arr}

