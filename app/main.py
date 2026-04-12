from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from statistics import mean
from typing import Deque, Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


class SensorEvent(BaseModel):
    student_id: str = Field(..., min_length=1)
    timestamp: datetime
    gaze_focus: float = Field(..., ge=0.0, le=1.0)
    head_motion: float = Field(..., ge=0.0, le=1.0)
    ambient_noise: float = Field(..., ge=0.0, le=1.0)
    blink_rate: float = Field(..., ge=0.0, le=80.0)


@dataclass
class ScoredEvent:
    student_id: str
    timestamp: str
    gaze_focus: float
    head_motion: float
    ambient_noise: float
    blink_rate: float
    attention_score: float
    attention_band: str


class ClassroomStore:
    def __init__(self, max_events: int = 1000) -> None:
        self.events: Deque[ScoredEvent] = deque(maxlen=max_events)
        self.latest_by_student: Dict[str, ScoredEvent] = {}

    def add(self, event: ScoredEvent) -> None:
        self.events.append(event)
        self.latest_by_student[event.student_id] = event

    def snapshot(self) -> dict:
        students = list(self.latest_by_student.values())
        avg = mean([s.attention_score for s in students]) if students else 0.0
        low_count = len([s for s in students if s.attention_band == "low"])
        return {
            "last_updated": datetime.now(tz=timezone.utc).isoformat(),
            "student_count": len(students),
            "average_attention": round(avg, 2),
            "low_attention_students": low_count,
            "students": [asdict(s) for s in sorted(students, key=lambda x: x.student_id)],
        }


class ConnectionManager:
    def __init__(self) -> None:
        self.connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def broadcast(self, payload: dict) -> None:
        stale: List[WebSocket] = []
        for conn in self.connections:
            try:
                await conn.send_json(payload)
            except Exception:
                stale.append(conn)
        for conn in stale:
            self.disconnect(conn)


app = FastAPI(title="Smart Classroom Attention Monitor", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/web", StaticFiles(directory="web"), name="web")

store = ClassroomStore()
ws_manager = ConnectionManager()


def compute_attention_score(event: SensorEvent) -> float:
    # High gaze helps attention, while motion/noise/blink spikes reduce it.
    blink_penalty = min(max((event.blink_rate - 16.0) / 32.0, 0.0), 1.0)
    weighted = (
        0.55 * event.gaze_focus
        + 0.2 * (1.0 - event.head_motion)
        + 0.15 * (1.0 - event.ambient_noise)
        + 0.1 * (1.0 - blink_penalty)
    )
    return round(max(0.0, min(1.0, weighted)) * 100.0, 2)


def attention_band(score: float) -> str:
    if score >= 75:
        return "high"
    if score >= 50:
        return "medium"
    return "low"


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse("web/index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/ingest")
async def ingest(event: SensorEvent) -> dict:
    score = compute_attention_score(event)
    scored = ScoredEvent(
        student_id=event.student_id,
        timestamp=event.timestamp.astimezone(timezone.utc).isoformat(),
        gaze_focus=event.gaze_focus,
        head_motion=event.head_motion,
        ambient_noise=event.ambient_noise,
        blink_rate=event.blink_rate,
        attention_score=score,
        attention_band=attention_band(score),
    )
    store.add(scored)
    snap = store.snapshot()
    await ws_manager.broadcast(snap)
    return {"accepted": True, "attention_score": score, "band": scored.attention_band}


@app.get("/api/snapshot")
def get_snapshot() -> dict:
    return store.snapshot()


@app.websocket("/ws/classroom")
async def ws_classroom(websocket: WebSocket) -> None:
    await ws_manager.connect(websocket)
    try:
        await websocket.send_json(store.snapshot())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)
