# Smart Classroom Attention Monitor

A complete IoT-style classroom attention monitoring demo with:
- FastAPI backend for ingest + scoring
- Web dashboard with live updates
- Simulator that publishes synthetic student attention telemetry
- Optional webcam publisher for live laptop camera tests

## Architecture
- `app/main.py`: API server, scoring logic, WebSocket broadcast
- `simulator/publisher.py`: synthetic sensor publisher
- `web/`: static dashboard

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the API server:

```powershell
uvicorn app.main:app --reload --port 8000
```

4. In another terminal, run the simulator:

```powershell
python simulator/publisher.py
```

Optional: run webcam publisher instead of synthetic simulation:

```powershell
python simulator/camera_publisher.py --preview
```

5. Open dashboard:

- http://127.0.0.1:8000

## Data Model
Each event contains:
- `student_id`
- `timestamp`
- `gaze_focus` (0-1)
- `head_motion` (0-1)
- `ambient_noise` (0-1)
- `blink_rate` (blinks/min)

Backend computes:
- `attention_score` (0-100)
- `attention_band` (`high`, `medium`, `low`)

## Notes
- This is a simulation for experimentation and teaching.
- Replace simulator input with real sensor data to integrate hardware.
- The webcam publisher uses simple computer vision heuristics as a starter baseline, not a production model.
