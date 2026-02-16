# NOEMA Project Structure

Project root:

`noema/`

## Top-level directories

- `backend/`
  - FastAPI app, API endpoints, inference logic
  - Domain routing for model selection (`rocket` / `spacesuit`)
  - Voice analysis pipeline

- `frontend/`
  - Single-page web interface (`index.html`)
  - UI for Satellite, Rocket, Spacesuit, and Voice modes
  - 3D visualization and charts

- `rocket yolo/`
  - Rocket dataset and training script
  - YOLO run artifacts/checkpoints

- `spacesuit damage detection.v1i.yolov8-obb/`
  - Spacesuit dataset/checkpoints used by spacesuit visual analysis

- `training/`
  - Additional training-related assets/materials

## Top-level files

- `RUN_INSTRUCTIONS.md` - run guide
- `PROJECT_DESCRIPTION.md` - project overview
- `PROJECT_STRUCTURE.md` - this file
- `yolov8n-obb.pt` - base YOLO weights file
