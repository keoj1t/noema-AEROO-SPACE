from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .acoustic_anomaly import compute_acoustic_anomaly
from .audio_ingest import audio_bytes_to_mono16k_float32
from .fusion import fuse_signals
from .recommendations import build_explanation, build_recommendations
from .schema import VoiceAnalysisResponse
from .stt_whisper import transcribe_with_whisper
from .voice_state import compute_stress_timeline


@dataclass
class VoiceAnalyzeConfig:
    audio_type: str = "generic"
    report_mode: str = "full"


def analyze_voice_bytes(
    audio_bytes: bytes,
    audio_suffix: str = ".wav",
    baseline_bytes: Optional[bytes] = None,
    baseline_suffix: str = ".wav",
    config: Optional[VoiceAnalyzeConfig] = None,
) -> VoiceAnalysisResponse:
    cfg = config or VoiceAnalyzeConfig()

    waveform, sr, normalized_audio_path = audio_bytes_to_mono16k_float32(audio_bytes, suffix=audio_suffix)
    duration_seconds = float(len(waveform) / sr) if sr > 0 else 0.0

    baseline_waveform = None
    if baseline_bytes:
        baseline_waveform, _bsr, _bpath = audio_bytes_to_mono16k_float32(baseline_bytes, suffix=baseline_suffix)

    stt = transcribe_with_whisper(normalized_audio_path)
    transcript_text = str(stt.get("text", "") or "").strip()
    transcript_segments = stt.get("segments", []) or []

    stress = compute_stress_timeline(waveform=waveform, sr=sr, baseline_waveform=baseline_waveform)
    anomaly = compute_acoustic_anomaly(waveform=waveform, sr=sr, baseline_waveform=baseline_waveform)
    fused = fuse_signals(
        stress_scores=np.asarray(stress.get("scores", []), dtype=np.float32),
        anomaly_scores=np.asarray(anomaly.get("scores", []), dtype=np.float32),
    )

    events = list(anomaly.get("events", [])) + list(fused.get("events", []))
    events = sorted(events, key=lambda x: (float(x.get("start", 0.0)), -float(x.get("score", 0.0))))

    recs = build_recommendations(fused["summary"], events)
    explanation = build_explanation(fused["summary"], events, transcript_text)

    summary = {
        "severity": fused["summary"]["severity"],
        "confidence": fused["summary"]["confidence"],
        "primary_risk_type": fused["summary"]["primary_risk_type"],
        "duration_seconds": duration_seconds,
        "transcript_text": transcript_text,
        "stress_peak": fused["summary"]["stress_peak"],
        "anomaly_peak": fused["summary"]["anomaly_peak"],
        "fused_peak": fused["summary"]["fused_peak"],
    }

    # Keep report_mode switch open for future compact/full behavior.
    if cfg.report_mode.lower() == "compact":
        transcript_segments = transcript_segments[: min(8, len(transcript_segments))]
        events = events[: min(8, len(events))]

    return VoiceAnalysisResponse(
        summary=summary,
        events=events,
        timeline=fused["timeline"],
        transcript_segments=transcript_segments,
        recommendations=recs,
        explanation_text=explanation,
    )

