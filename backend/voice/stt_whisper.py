from typing import Dict, List


def transcribe_with_whisper(audio_path: str) -> Dict[str, object]:
    """
    Returns:
    {
      "text": str,
      "segments": [{"start": float, "end": float, "text": str, "confidence": float|None}]
    }
    """
    # Preferred backend: faster-whisper
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, _info = model.transcribe(audio_path, vad_filter=True)
        segs: List[Dict[str, object]] = []
        text_parts: List[str] = []
        for s in segments:
            t = (s.text or "").strip()
            if not t:
                continue
            segs.append(
                {
                    "start": float(s.start),
                    "end": float(s.end),
                    "text": t,
                    "confidence": None,
                }
            )
            text_parts.append(t)
        return {"text": " ".join(text_parts).strip(), "segments": segs}
    except Exception:
        pass

    # Fallback: openai-whisper package if present
    try:
        import whisper

        model = whisper.load_model("base")
        result = model.transcribe(audio_path, fp16=False)
        segs: List[Dict[str, object]] = []
        text_parts: List[str] = []
        for s in result.get("segments", []):
            t = (s.get("text") or "").strip()
            if not t:
                continue
            segs.append(
                {
                    "start": float(s.get("start", 0.0)),
                    "end": float(s.get("end", 0.0)),
                    "text": t,
                    "confidence": None,
                }
            )
            text_parts.append(t)
        return {"text": " ".join(text_parts).strip(), "segments": segs}
    except Exception:
        return {"text": "", "segments": []}

