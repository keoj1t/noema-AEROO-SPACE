from typing import Dict, List

import numpy as np


def _severity_from_score(score: float) -> str:
    if score >= 0.8:
        return "critical"
    if score >= 0.6:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def fuse_signals(
    stress_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    stress_threshold: float = 0.7,
    anomaly_threshold: float = 0.75,
) -> Dict[str, object]:
    n = max(len(stress_scores), len(anomaly_scores))
    if n == 0:
        return {
            "timeline": [],
            "events": [],
            "summary": {
                "severity": "low",
                "confidence": 0.2,
                "primary_risk_type": "none",
                "fused_peak": 0.0,
            },
        }

    s = np.zeros(n, dtype=np.float32)
    a = np.zeros(n, dtype=np.float32)
    s[: len(stress_scores)] = stress_scores
    a[: len(anomaly_scores)] = anomaly_scores
    fused = np.clip(0.55 * s + 0.45 * a, 0.0, 1.0)

    timeline = []
    events: List[Dict[str, object]] = []
    for i in range(n):
        timeline.append(
            {
                "t": float(i),
                "stress_score": float(s[i]),
                "anomaly_score": float(a[i]),
                "fused_score": float(fused[i]),
            }
        )
        if s[i] > stress_threshold and a[i] > anomaly_threshold:
            events.append(
                {
                    "start": float(i),
                    "end": float(i + 2.0),
                    "event_type": "combined_stress_anomaly",
                    "score": float(fused[i]),
                    "label": "Combined Voice Risk",
                    "details": "Stress and acoustic anomaly overlap in the same time window.",
                }
            )

    stress_peak = float(np.max(s)) if n else 0.0
    anomaly_peak = float(np.max(a)) if n else 0.0
    fused_peak = float(np.max(fused)) if n else 0.0

    if stress_peak >= anomaly_peak and stress_peak > 0.5:
        primary = "stress"
    elif anomaly_peak > 0.5:
        primary = "acoustic_anomaly"
    else:
        primary = "none"

    summary = {
        "severity": _severity_from_score(fused_peak),
        "confidence": float(np.clip(0.35 + 0.65 * fused_peak, 0.0, 1.0)),
        "primary_risk_type": primary,
        "stress_peak": stress_peak,
        "anomaly_peak": anomaly_peak,
        "fused_peak": fused_peak,
    }
    return {"timeline": timeline, "events": events, "summary": summary}

