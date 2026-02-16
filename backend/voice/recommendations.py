from typing import Dict, List


def build_recommendations(summary: Dict[str, object], events: List[Dict[str, object]]) -> List[str]:
    recs: List[str] = []
    severity = str(summary.get("severity", "low"))
    primary = str(summary.get("primary_risk_type", "none"))

    if severity in {"critical", "high"}:
        recs.append("Escalate to human operator immediately and re-check mission-critical communication protocol.")
        recs.append("Record a second confirmation sample in controlled conditions within 1-2 minutes.")
    else:
        recs.append("Continue monitoring voice channel and repeat diagnostics if tone or clarity changes.")

    if primary == "stress":
        recs.append("Ask speaker to slow down and confirm checklist items with closed-loop responses.")
    elif primary == "acoustic_anomaly":
        recs.append("Inspect microphone channel for clipping, packet loss, or environmental interference.")

    if any(e.get("event_type") == "combined_stress_anomaly" for e in events):
        recs.append("Combined stress+acoustic anomaly detected: prioritize communication safety and redundancy channel.")

    if not recs:
        recs.append("No immediate action required.")
    return recs


def build_explanation(summary: Dict[str, object], events: List[Dict[str, object]], transcript: str) -> str:
    sev = str(summary.get("severity", "low"))
    primary = str(summary.get("primary_risk_type", "none"))
    cnt = len(events)
    if not transcript:
        tx = "Transcript unavailable."
    else:
        tx = f"Transcript extracted ({len(transcript.split())} words)."
    return (
        f"Voice diagnostics completed with {sev} severity and primary risk type '{primary}'. "
        f"Detected {cnt} notable event(s) across 2-second windows. {tx} "
        "Stress score is derived from embedding deviation against baseline; anomaly score is derived from mel-spectrogram autoencoder reconstruction error."
    )

