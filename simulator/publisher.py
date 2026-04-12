from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone

import httpx


API_URL = "http://127.0.0.1:8000/api/ingest"
STUDENTS = [f"student-{i:02d}" for i in range(1, 21)]


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def generate_event(student_id: str) -> dict:
    base_attention = random.uniform(0.35, 0.95)
    gaze_focus = clamp(base_attention + random.uniform(-0.1, 0.1))
    head_motion = clamp(1.0 - base_attention + random.uniform(-0.2, 0.2))
    ambient_noise = clamp(random.uniform(0.05, 0.75))
    blink_rate = max(8.0, min(40.0, random.gauss(18.0 + (1.0 - base_attention) * 12, 3.0)))

    return {
        "student_id": student_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "gaze_focus": round(gaze_focus, 3),
        "head_motion": round(head_motion, 3),
        "ambient_noise": round(ambient_noise, 3),
        "blink_rate": round(blink_rate, 2),
    }


async def main() -> None:
    print("Publishing synthetic classroom telemetry. Press Ctrl+C to stop.")
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            batch = random.sample(STUDENTS, k=random.randint(8, 16))
            for sid in batch:
                event = generate_event(sid)
                try:
                    await client.post(API_URL, json=event)
                except httpx.HTTPError as exc:
                    print(f"post failed for {sid}: {exc}")
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Simulator stopped.")
