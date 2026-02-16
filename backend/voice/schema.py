from typing import List, Optional

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    confidence: Optional[float] = None


class VoiceEvent(BaseModel):
    start: float
    end: float
    event_type: str
    score: float = Field(ge=0.0, le=1.0)
    label: str
    details: str


class TimelinePoint(BaseModel):
    t: float
    stress_score: float = Field(ge=0.0, le=1.0)
    anomaly_score: float = Field(ge=0.0, le=1.0)
    fused_score: float = Field(ge=0.0, le=1.0)


class VoiceSummary(BaseModel):
    severity: str
    confidence: float = Field(ge=0.0, le=1.0)
    primary_risk_type: str
    duration_seconds: float
    transcript_text: str
    stress_peak: float = Field(ge=0.0, le=1.0)
    anomaly_peak: float = Field(ge=0.0, le=1.0)
    fused_peak: float = Field(ge=0.0, le=1.0)


class VoiceAnalysisResponse(BaseModel):
    status: str = "success"
    summary: VoiceSummary
    events: List[VoiceEvent]
    timeline: List[TimelinePoint]
    transcript_segments: List[TranscriptSegment]
    recommendations: List[str]
    explanation_text: str

