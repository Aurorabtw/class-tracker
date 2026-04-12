from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

import httpx

try:
    import cv2
except Exception:
    cv2 = None


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


class CameraAttentionEstimator:
    def __init__(self, cv2_module: Any) -> None:
        self.cv2 = cv2_module
        self.face_cascade = cv2_module.CascadeClassifier(
            cv2_module.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2_module.CascadeClassifier(
            cv2_module.data.haarcascades + "haarcascade_eye.xml"
        )
        self.prev_center: Optional[Tuple[float, float]] = None
        self.blink_rate_ema: float = 16.0

    def estimate(self, frame: Any) -> tuple[dict, Optional[tuple[int, int, int, int]]]:
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80))

        if len(faces) == 0:
            metrics = {
                "gaze_focus": 0.25,
                "head_motion": 0.75,
                "blink_rate": round(self._smooth_blink(26.0), 2),
            }
            self.prev_center = None
            return metrics, None

        # Use largest face in frame.
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        frame_h, frame_w = gray.shape

        cx = x + (w / 2.0)
        cy = y + (h / 2.0)
        area_ratio = (w * h) / float(frame_w * frame_h)

        nx = (cx / frame_w) - 0.5
        ny = (cy / frame_h) - 0.5
        center_dist = (nx * nx + ny * ny) ** 0.5

        gaze_focus = clamp(0.35 + (area_ratio * 1.8) - (center_dist * 0.9))

        head_motion = 0.1
        if self.prev_center is not None:
            dx = cx - self.prev_center[0]
            dy = cy - self.prev_center[1]
            pixel_motion = (dx * dx + dy * dy) ** 0.5
            head_motion = clamp(pixel_motion / max(24.0, frame_w * 0.08))
        self.prev_center = (cx, cy)

        roi_gray = gray[y : y + h, x : x + w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
        eye_visibility = min(len(eyes), 2) / 2.0
        blink_proxy = 1.0 - eye_visibility

        raw_blink_rate = 12.0 + (blink_proxy * 18.0) + (head_motion * 10.0)
        blink_rate = self._smooth_blink(raw_blink_rate)

        metrics = {
            "gaze_focus": round(gaze_focus, 3),
            "head_motion": round(head_motion, 3),
            "blink_rate": round(blink_rate, 2),
        }
        return metrics, (x, y, w, h)

    def _smooth_blink(self, value: float) -> float:
        self.blink_rate_ema = (0.75 * self.blink_rate_ema) + (0.25 * value)
        return self.blink_rate_ema


async def run(args: argparse.Namespace) -> None:
    if cv2 is None:
        print("OpenCV is not installed. Run: python -m pip install opencv-python")
        return

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Could not open camera index {args.camera_index}.")
        return

    estimator = CameraAttentionEstimator(cv2)
    interval = max(0.15, args.interval)

    print("Camera publisher started. Press 'q' in preview window to stop.")
    print(f"Sending events to {args.api_url} every {interval:.2f}s")

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame read failed.")
                    await asyncio.sleep(interval)
                    continue

                metrics, face_bbox = estimator.estimate(frame)
                payload = {
                    "student_id": args.student_id,
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    "gaze_focus": metrics["gaze_focus"],
                    "head_motion": metrics["head_motion"],
                    "ambient_noise": args.ambient_noise,
                    "blink_rate": metrics["blink_rate"],
                }

                try:
                    await client.post(args.api_url, json=payload)
                except httpx.HTTPError as exc:
                    print(f"POST failed: {exc}")

                if args.preview:
                    display = frame.copy()
                    if face_bbox:
                        x, y, w, h = face_bbox
                        cv2.rectangle(display, (x, y), (x + w, y + h), (80, 220, 120), 2)
                    overlay = (
                        f"focus={metrics['gaze_focus']:.2f} "
                        f"motion={metrics['head_motion']:.2f} "
                        f"blink={metrics['blink_rate']:.1f}/min"
                    )
                    cv2.putText(
                        display,
                        overlay,
                        (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Classroom Camera Publisher", display)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break

                await asyncio.sleep(interval)
        finally:
            cap.release()
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish webcam-derived attention metrics to backend.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/api/ingest")
    parser.add_argument("--student-id", default="camera-student-01")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.6, help="Seconds between payloads.")
    parser.add_argument("--ambient-noise", type=float, default=0.2)
    parser.add_argument("--preview", action="store_true", help="Show camera preview with overlay.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("Camera publisher stopped.")
