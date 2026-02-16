# NOEMA Run Instructions

This guide is for third parties who clone/download the project and need to run it locally.

## 1) Prerequisites

- Python 3.10+
- FFmpeg in system `PATH` (required for the Voice module)

Check FFmpeg:

```bash
ffmpeg -version
```

## 2) Open project root

```bash
cd noema
```

## 3) Install dependencies

```bash
pip install -r backend/requirements.txt
```

## 4) Start the server

```bash
python backend/main.py
```

## 5) Open the app

- App: `http://localhost:8000`
- API Docs: `http://localhost:8000/api/docs`

## 6) Stop the server

Press `Ctrl + C` in the same terminal.
