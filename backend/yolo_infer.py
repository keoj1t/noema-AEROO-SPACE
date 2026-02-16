"""
YOLO VISION-BASED ANOMALY DETECTION

Purpose:
- Secondary anomaly signal from rendered 3D visualization frames
- Detects visual inconsistencies in system state visualizations
- Supports and explains time-series anomalies

Important:
- Input images are NOT real satellite photos
- Images are 3D schematic visualizations for explainability
- Output supplements but does NOT override time-series analysis

Input:
- PNG/JPG frames from 3D visualization layer

Output:
- Bounding boxes of anomalies
- Confidence scores
- High-level anomaly flag (boolean)

Constraints:
- Inference only
- No online training
- No database
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import cv2

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def _install_numpy_legacy_aliases() -> None:
    """
    Some YOLO checkpoints were serialized with numpy._core module paths (NumPy 2).
    Current runtime may expose numpy.core only (NumPy 1.x), so we add aliases.
    """
    try:
        import numpy.core as core  # type: ignore
        import numpy.core.multiarray as multiarray  # type: ignore
        import numpy.core.umath as umath  # type: ignore
        import numpy.core.numerictypes as numerictypes  # type: ignore
        import numpy.core._multiarray_umath as _multiarray_umath  # type: ignore

        sys.modules.setdefault("numpy._core", core)
        sys.modules.setdefault("numpy._core.multiarray", multiarray)
        sys.modules.setdefault("numpy._core.umath", umath)
        sys.modules.setdefault("numpy._core.numerictypes", numerictypes)
        sys.modules.setdefault("numpy._core._multiarray_umath", _multiarray_umath)
    except Exception:
        # Keep startup resilient; loader will report actual error if loading still fails.
        pass

class YOLOAnomalyDetector:
    """
    Lightweight visual anomaly detector.
    Analyzes rendered 3D visualization frames for visual inconsistencies.
    Uses heuristic color/brightness patterns without pretrained YOLO.
    """
    
    _cached_model = None
    _cached_model_path = None
    _cached_load_error = None
    _cached_models_by_path: Dict[str, Any] = {}

    def __init__(self, model_path: Optional[str] = None):
        self.anomaly_threshold = 0.5
        self.model_path = self._resolve_model_path(model_path)
        self.backup_model_paths = self._resolve_backup_model_paths(self.model_path)
        self.model = None
        self.model_loaded = False
        self.load_error = None
        self._load_model()

    def _candidate_model_paths(self, explicit_path: Optional[str] = None) -> List[Path]:
        if explicit_path:
            p = Path(explicit_path).expanduser().resolve()
            return [p] if p.exists() else []

        env_path = os.getenv("ROCKET_YOLO_MODEL") or os.getenv("SPACESUIT_YOLO_MODEL")
        if env_path:
            p = Path(env_path).expanduser().resolve()
            if p.exists():
                return [p]

        backend_dir = Path(__file__).resolve().parent
        app_root = backend_dir.parent
        workspace_root = app_root.parent

        candidates = [
            app_root / "rocket yolo" / "runs" / "detect" / "train" / "weights" / "best.pt",
            app_root / "rocket yolo" / "runs" / "detect" / "train" / "weights" / "last.pt",
            workspace_root / "rocket yolo" / "runs" / "detect" / "train" / "weights" / "best.pt",
            workspace_root / "rocket yolo" / "runs" / "detect" / "train" / "weights" / "last.pt",
            app_root / "spacesuit damage detection.v1i.yolov8-obb" / "runs" / "detect" / "train4_obb_retrain" / "weights" / "best.pt",
            app_root / "spacesuit damage detection.v1i.yolov8-obb" / "runs" / "detect" / "train4_obb_retrain" / "weights" / "last.pt",
            app_root / "spacesuit damage detection.v1i.yolov8-obb" / "runs" / "detect" / "train3" / "weights" / "last.pt",
            app_root / "spacesuit" / "train3" / "weights" / "last.pt",
            workspace_root / "spacesuit" / "train3" / "weights" / "last.pt",
            app_root / "spacesuit" / "train3" / "weights" / "best.pt",
            app_root / "spacesuit" / "train3" / "best.pt",
            app_root / "spacesuit damage detection.v1i.yolov8-obb" / "runs" / "detect" / "train3" / "weights" / "best.pt",
            workspace_root / "spacesuit" / "train3" / "weights" / "best.pt",
            workspace_root / "spacesuit" / "train3" / "best.pt",
            app_root / "SpaceSuit" / "train3" / "weights" / "best.pt",
            workspace_root / "SpaceSuit" / "train3" / "weights" / "best.pt",
        ]
        out: List[Path] = []
        for candidate in candidates:
            if candidate.exists():
                resolved = candidate.resolve()
                if resolved not in out:
                    out.append(resolved)
        return out

    def _resolve_model_path(self, explicit_path: Optional[str] = None) -> Optional[Path]:
        paths = self._candidate_model_paths(explicit_path)
        return paths[0] if paths else None

    def _resolve_backup_model_paths(self, selected: Optional[Path], explicit_path: Optional[str] = None) -> List[Path]:
        paths = self._candidate_model_paths(explicit_path)
        if not selected:
            return paths
        return [p for p in paths if str(p) != str(selected)]

    def _load_model(self) -> None:
        if self.model_path is None:
            self.load_error = "YOLO checkpoint not found (rocket yolo/runs/detect/train/weights)"
            return

        if YOLO is None:
            self.load_error = "ultralytics is not installed"
            return

        model, err = self._load_model_by_path(self.model_path)
        if model is not None:
            self.model = model
            self.model_loaded = True
            self.load_error = None
            return

        self.model_loaded = False
        self.model = None
        self.load_error = err or "unknown model load error"

    def _load_model_by_path(self, path: Path):
        path_str = str(path)
        cached = self.__class__._cached_models_by_path.get(path_str)
        if cached is not None:
            return cached, None

        try:
            # Compatibility shim for checkpoints saved with NumPy 2 module paths.
            _install_numpy_legacy_aliases()
            model = YOLO(path_str)
            self.__class__._cached_models_by_path[path_str] = model
            self.__class__._cached_model = model
            self.__class__._cached_model_path = path_str
            self.__class__._cached_load_error = None
            return model, None
        except Exception as e:
            self.__class__._cached_load_error = str(e)
            return None, str(e)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_loaded": bool(self.model_loaded),
            "model_path": str(self.model_path) if self.model_path else None,
            "backup_models": [str(p) for p in self.backup_model_paths],
            "load_error": self.load_error,
            "ultralytics_available": YOLO is not None,
        }

    @staticmethod
    def _safe_name(names: Any, cls_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        if isinstance(names, list) and 0 <= cls_id < len(names):
            return str(names[cls_id])
        return str(cls_id)

    @staticmethod
    def _dedupe_detections(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique: Dict[tuple, Dict[str, Any]] = {}
        for det in detections:
            bbox = det.get("bbox", {})
            key = (
                det.get("class_id"),
                round(float(bbox.get("x1", 0.0)), 1),
                round(float(bbox.get("y1", 0.0)), 1),
                round(float(bbox.get("x2", 0.0)), 1),
                round(float(bbox.get("y2", 0.0)), 1),
            )
            if key not in unique or float(det.get("confidence", 0.0)) > float(unique[key].get("confidence", 0.0)):
                unique[key] = det
        return sorted(unique.values(), key=lambda d: float(d.get("confidence", 0.0)), reverse=True)

    def _extract_detections_from_result(self, result: Any) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        names = result.names if hasattr(result, "names") else {}

        # OBB models (yolov8-obb) expose detections under result.obb.
        obb = getattr(result, "obb", None)
        if obb is not None and hasattr(obb, "cls") and hasattr(obb, "conf"):
            try:
                count = len(obb.cls)
            except Exception:
                count = 0
            for i in range(count):
                cls_id = int(obb.cls[i].item())
                conf = float(obb.conf[i].item())
                polygon = []
                x1 = y1 = x2 = y2 = None
                if hasattr(obb, "xyxyxyxy"):
                    try:
                        points = obb.xyxyxyxy[i].tolist()
                        polygon = [[float(p[0]), float(p[1])] for p in points]
                        xs = [p[0] for p in polygon]
                        ys = [p[1] for p in polygon]
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    except Exception:
                        polygon = []
                if x1 is None and hasattr(obb, "xyxy"):
                    try:
                        x1, y1, x2, y2 = [float(v) for v in obb.xyxy[i].tolist()]
                    except Exception:
                        continue

                detections.append(
                    {
                        "class": self._safe_name(names, cls_id),
                        "class_id": cls_id,
                        "confidence": conf,
                        "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                        "polygon": polygon,
                        "is_obb": True,
                    }
                )

        # Standard axis-aligned boxes.
        boxes = getattr(result, "boxes", None)
        if boxes is not None and hasattr(boxes, "cls") and hasattr(boxes, "conf"):
            try:
                count = len(boxes.cls)
            except Exception:
                count = 0
            for i in range(count):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
                detections.append(
                    {
                        "class": self._safe_name(names, cls_id),
                        "class_id": cls_id,
                        "confidence": conf,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "is_obb": False,
                    }
                )

        return detections

    @staticmethod
    def _filter_detections(
        detections: List[Dict[str, Any]],
        min_conf: float = 0.05,
        max_total: int = 9,
        max_per_class: int = 4
    ) -> List[Dict[str, Any]]:
        if not detections:
            return []

        # Adaptive confidence floor: keep only detections close to strongest prediction.
        top_conf = max(float(d.get("confidence", 0.0)) for d in detections)
        adaptive_conf = max(float(min_conf), top_conf * 0.38)

        # Estimate image extent from boxes to drop absurdly large/full-frame boxes.
        max_x = max(float(d.get("bbox", {}).get("x2", 0.0)) for d in detections) + 1.0
        max_y = max(float(d.get("bbox", {}).get("y2", 0.0)) for d in detections) + 1.0
        frame_area = max_x * max_y

        cleaned: List[Dict[str, Any]] = []
        for det in detections:
            conf = float(det.get("confidence", 0.0))
            if conf < adaptive_conf:
                continue
            bbox = det.get("bbox", {})
            x1, y1 = float(bbox.get("x1", 0.0)), float(bbox.get("y1", 0.0))
            x2, y2 = float(bbox.get("x2", 0.0)), float(bbox.get("y2", 0.0))
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w < 8 or h < 8:
                continue
            area = w * h
            if frame_area > 0:
                area_ratio = area / frame_area
                if area_ratio < 0.00035 or area_ratio > 0.35:
                    continue

                # Suppress recurrent false positives in lower peripheral suit zones
                # (boots/connectors often look like defects to the model).
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                cx_ratio = cx / max_x
                cy_ratio = cy / max_y
                in_lower_band = cy_ratio > 0.72
                in_side_band = (cx_ratio < 0.34) or (cx_ratio > 0.66)
                if in_lower_band and in_side_band and area_ratio < 0.11 and conf < 0.88:
                    continue

            cleaned.append(det)

        if not cleaned:
            return []

        # Keep strongest per class first, then global cap.
        per_class_counts: Dict[str, int] = {}
        kept: List[Dict[str, Any]] = []
        for det in sorted(cleaned, key=lambda d: float(d.get("confidence", 0.0)), reverse=True):
            cls = str(det.get("class", "unknown"))
            cnt = per_class_counts.get(cls, 0)
            if cnt >= int(max_per_class):
                continue
            per_class_counts[cls] = cnt + 1
            kept.append(det)
            if len(kept) >= int(max_total):
                break
        return kept

    def _heuristic_damage_regions(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Fallback detector used only when YOLO returns no objects.
        Finds suspicious high-contrast elongated regions (typical cracks/tears).
        """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 80, 200)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape[:2]
        min_area = max(100, int(0.00015 * w * h))
        max_area = int(0.08 * w * h)

        out: List[Dict[str, Any]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = max(bw, bh) / max(1, min(bw, bh))
            # Favor elongated or compact high-detail regions
            if aspect < 1.2 and area < min_area * 3:
                continue

            roi = gray[y:y + bh, x:x + bw]
            if roi.size == 0:
                continue
            stdv = float(np.std(roi))
            mean_int = float(np.mean(roi))
            dark_ratio = float(np.mean(roi < 70))
            edge_ratio = float(np.mean(edges[y:y + bh, x:x + bw] > 0))

            # Reject clean patches, logos and smooth seams.
            if dark_ratio < 0.03 or edge_ratio < 0.05:
                continue
            if mean_int > 190 and dark_ratio < 0.08:
                continue

            conf = min(0.75, max(0.25, (stdv / 72.0) + dark_ratio * 0.7 + edge_ratio * 0.4))
            if conf < 0.42:
                continue

            out.append(
                {
                    "class": "damage_candidate",
                    "class_id": -1,
                    "confidence": float(conf),
                    "bbox": {
                        "x1": float(x),
                        "y1": float(y),
                        "x2": float(x + bw),
                        "y2": float(y + bh),
                    },
                    "is_obb": False,
                    "source": "heuristic_fallback",
                }
            )

        if not out:
            return []

        # NMS to reduce overlapping boxes.
        boxes_xywh = []
        scores = []
        for det in out:
            b = det["bbox"]
            x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
            boxes_xywh.append([x1, y1, max(1, x2 - x1), max(1, y2 - y1)])
            scores.append(float(det["confidence"]))

        keep = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=0.42, nms_threshold=0.35)
        kept: List[Dict[str, Any]] = []
        if len(keep) > 0:
            for idx in np.array(keep).reshape(-1).tolist():
                kept.append(out[idx])

        # Keep only strongest candidates to avoid clutter
        kept = sorted(kept, key=lambda d: d["confidence"], reverse=True)[:5]
        return self._dedupe_detections(kept)

    def _run_model_predict(
        self,
        model: Any,
        image_path: str,
        conf_threshold: float
    ) -> List[Dict[str, Any]]:
        base_conf = float(np.clip(conf_threshold, 0.005, 0.95))
        # Multi-pass sensitivity profile for hard-to-detect suit defects.
        pass_confs = [base_conf, max(0.02, base_conf * 0.7), max(0.01, base_conf * 0.5)]

        all_detections: List[Dict[str, Any]] = []
        for idx, pass_conf in enumerate(pass_confs):
            results = model.predict(
                source=image_path,
                conf=pass_conf,
                iou=0.6,
                imgsz=1600,
                augment=False,
                agnostic_nms=False,
                max_det=300,
                verbose=False,
            )
            if not results:
                continue
            for result in results:
                all_detections.extend(self._extract_detections_from_result(result))
        return self._dedupe_detections(all_detections)

    def analyze_image_path(
        self,
        image_path: str,
        conf_threshold: float = 0.01,
        use_fallback: bool = False
    ) -> List[Dict[str, Any]]:
        if not self.model_loaded or self.model is None:
            return []

        try:
            yolo_detections = self._filter_detections(
                self._run_model_predict(self.model, image_path, conf_threshold)
            )
            if yolo_detections:
                for det in yolo_detections:
                    det.setdefault("source", "yolo")
                return yolo_detections

            # Auto low-confidence pass (YOLO classes only) if strict pass produced nothing.
            low_conf_detections = self._filter_detections(
                self._run_model_predict(self.model, image_path, 0.01),
                min_conf=0.01,
                max_total=6,
                max_per_class=3,
            )
            if low_conf_detections:
                for det in low_conf_detections:
                    det.setdefault("source", "yolo_lowconf")
                return low_conf_detections

            # Try backup checkpoints if primary model returned nothing.
            for backup_path in self.backup_model_paths:
                backup_model, _ = self._load_model_by_path(backup_path)
                if backup_model is None:
                    continue
                backup_detections = self._filter_detections(
                    self._run_model_predict(backup_model, image_path, conf_threshold)
                )
                if backup_detections:
                    for det in backup_detections:
                        det.setdefault("source", "yolo_backup")
                        det["model"] = backup_path.name
                    return backup_detections

            # Optional fallback: off by default to avoid false positives and fake classes.
            if use_fallback:
                return self._heuristic_damage_regions(image_path)
            return []
        except Exception:
            return []
        
    def analyze_frame(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single visualization frame for visual anomalies.
        
        Args:
            frame_data: Dict with frame info from 3D visualization
                {
                    'health': float (0-100),
                    'anomaly_score': float (0-1),
                    'timestamp': str,
                    'satellite_color': str (hex),
                    'glow_intensity': float
                }
        
        Returns:
            Detection results with bounding boxes and confidence
        """
        results = {
            'detections': [],
            'anomaly_detected': False,
            'confidence': 0.0,
            'visual_signals': []
        }
        
        # Extract visual indicators from frame data
        health = frame_data.get('health', 100)
        anomaly_score = frame_data.get('anomaly_score', 0.0)
        glow_intensity = frame_data.get('glow_intensity', 0.0)
        
        # Visual anomaly indicators
        visual_anomaly_score = 0.0
        
        # Signal 1: Health-based color change
        if health < 40:
            # Red indicator - critical state
            visual_anomaly_score += 0.7
            results['visual_signals'].append({
                'type': 'critical_health',
                'severity': 'high',
                'value': health
            })
        elif health < 70:
            # Yellow indicator - degradation
            visual_anomaly_score += 0.4
            results['visual_signals'].append({
                'type': 'degraded_health',
                'severity': 'medium',
                'value': health
            })
        
        # Signal 2: Glow intensity (indicates anomaly level)
        if glow_intensity > 0.6:
            visual_anomaly_score += 0.5
            results['visual_signals'].append({
                'type': 'high_glow',
                'severity': 'high',
                'intensity': glow_intensity
            })
        elif glow_intensity > 0.3:
            visual_anomaly_score += 0.2
            results['visual_signals'].append({
                'type': 'moderate_glow',
                'severity': 'low',
                'intensity': glow_intensity
            })
        
        # Signal 3: Temporal consistency (anomaly score variance)
        if anomaly_score > 0.7:
            visual_anomaly_score += 0.6
            results['visual_signals'].append({
                'type': 'high_anomaly_score',
                'severity': 'high',
                'score': anomaly_score
            })
        
        # Normalize confidence
        visual_anomaly_score = np.clip(visual_anomaly_score / 3.0, 0, 1.0)
        
        results['confidence'] = float(visual_anomaly_score)
        results['anomaly_detected'] = visual_anomaly_score > self.anomaly_threshold
        
        # Generate bounding box for satellite region if anomaly detected
        if results['anomaly_detected']:
            results['detections'].append({
                'class': 'satellite_anomaly',
                'confidence': float(visual_anomaly_score),
                'bbox': {
                    'x': 0.25,  # Normalized coordinates
                    'y': 0.25,
                    'width': 0.5,
                    'height': 0.5
                },
                'visual_indicator': 'glow_halo' if glow_intensity > 0.5 else 'color_change'
            })
        
        # Ensure all values are Python native types
        results['anomaly_detected'] = bool(results['anomaly_detected'])
        results['confidence'] = float(results['confidence'])
        
        return results
    
    def fuse_with_temporal(self, 
                          temporal_anomaly: float,
                          visual_anomaly: float) -> Dict[str, Any]:
        """
        Fuse temporal (Chronos) and visual (YOLO) anomaly signals.
        
        Args:
            temporal_anomaly: Anomaly score from time-series analysis (0-1)
            visual_anomaly: Anomaly score from visual analysis (0-1)
        
        Returns:
            Fused result with interpretation
        """
        # Simple fusion: average + confidence weighting
        fused_score = (temporal_anomaly + visual_anomaly) / 2.0
        
        fusion_result = {
            'temporal_score': float(temporal_anomaly),
            'visual_score': float(visual_anomaly),
            'fused_score': float(fused_score),
            'agreement': 'high' if abs(temporal_anomaly - visual_anomaly) < 0.2 else 'low',
            'interpretation': ''
        }
        
        # Interpret fusion result
        if temporal_anomaly > 0.6 and visual_anomaly > 0.6:
            fusion_result['interpretation'] = 'STRONG_AGREEMENT: Temporal and visual signals both indicate anomaly'
            fusion_result['confidence'] = 'HIGH'
        elif temporal_anomaly > 0.6 and visual_anomaly < 0.3:
            fusion_result['interpretation'] = 'TEMPORAL_ONLY: Time-series anomaly without visual confirmation (latent failure)'
            fusion_result['confidence'] = 'MEDIUM'
        elif temporal_anomaly < 0.3 and visual_anomaly > 0.6:
            fusion_result['interpretation'] = 'VISUAL_ONLY: Visual anomaly without temporal confirmation (transient glitch)'
            fusion_result['confidence'] = 'MEDIUM'
        else:
            fusion_result['interpretation'] = 'NO_AGREEMENT: Both signals nominal or conflicting'
            fusion_result['confidence'] = 'LOW'
        
        # Ensure all values are native Python types
        fusion_result['temporal_score'] = float(fusion_result['temporal_score'])
        fusion_result['visual_score'] = float(fusion_result['visual_score'])
        fusion_result['fused_score'] = float(fusion_result['fused_score'])
        
        return fusion_result

def detect_visual_anomalies(frame_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Public interface for YOLO-based visual anomaly detection.
    
    Args:
        frame_data: Frame data from 3D visualization
    
    Returns:
        Detection results
    """
    detector = YOLOAnomalyDetector()
    return detector.analyze_frame(frame_data)
