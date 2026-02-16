# NOEMA Project Description

`NOEMA` is a local multi-domain diagnostics platform for aerospace-oriented analysis.

The application combines multiple AI-assisted workflows in one interface:

- `Satellite` domain: CSV telemetry diagnostics, anomaly timeline, and report generation.
- `Rocket` domain: image upload, YOLO-based defect detection, and 3D visualization mapping.
- `Spacesuit` domain: image upload, YOLO-based defect detection, and 3D visualization mapping.
- `Voice` domain: local audio diagnostics with event timeline, stress/anomaly indicators, and recommendations.

## Core idea

The system follows a black-box analysis approach: infer operational issues from observable signals (telemetry, images, audio) without requiring internal proprietary implementation details.

## Technology stack

- Backend: `FastAPI`, `Python`, `ONNX Runtime`, `Ultralytics YOLO`
- Frontend: `HTML`, `CSS`, `JavaScript`, `Three.js`, `Chart.js`
- Runtime model: single backend service that serves both API and frontend

## Intended use

- Local analysis environment for demos, prototyping, and diagnostics workflows
- Unified interface for multiple data modalities and domains
