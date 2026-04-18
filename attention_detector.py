# ============================================================
# AI Attention Detector
# Uses MediaPipe Face Landmarker (Tasks API, mediapipe 0.10+)
# to detect 478 facial landmarks and determine attention state.
# ============================================================

import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

# ── Model download (runs once) ───────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

if not os.path.exists(_MODEL_PATH):
    print("Downloading face_landmarker.task model (one-time) …")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("Model ready.")

# ── Initialise landmarker ────────────────────────────────────
_options = mp_vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
    num_faces=10,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
)
_landmarker = mp_vision.FaceLandmarker.create_from_options(_options)

# Left / right eye landmark indices (same 468-point map as FaceMesh)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]


def _calculate_EAR(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C) if C > 0 else 0.0


def _check_head_pose(landmarks, w, h):
    nose  = landmarks[1]
    l_eye = landmarks[33]
    r_eye = landmarks[263]

    nose_x       = nose.x  * w
    eye_center_x = (l_eye.x + r_eye.x) / 2 * w
    eye_width    = abs(r_eye.x - l_eye.x) * w

    offset_ratio = abs(nose_x - eye_center_x) / eye_width if eye_width > 0 else 1.0
    is_forward   = offset_ratio < 0.35

    nose_y       = nose.y  * h
    eye_center_y = (l_eye.y + r_eye.y) / 2 * h
    nod_ratio    = (nose_y - eye_center_y) / (h * 0.3)
    not_nodding  = nod_ratio < 1.5

    return is_forward and not_nodding


def analyze_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Could not load image", "attentive": 0,
                "distracted": 0, "total_faces": 0, "score": 0, "details": []}

    h, w   = img.shape[:2]
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
    result  = _landmarker.detect(mp_img)

    if not result.face_landmarks:
        return {"attentive": 0, "distracted": 0,
                "total_faces": 0, "score": 0, "details": []}

    attentive_count  = 0
    distracted_count = 0
    details          = []

    for face_idx, lm in enumerate(result.face_landmarks):
        left_ear  = _calculate_EAR(lm, LEFT_EYE,  w, h)
        right_ear = _calculate_EAR(lm, RIGHT_EYE, w, h)
        avg_ear   = (left_ear + right_ear) / 2
        eyes_open = avg_ear > 0.20

        facing_forward = _check_head_pose(lm, w, h)
        is_attentive   = eyes_open and facing_forward

        if is_attentive:
            attentive_count += 1
        else:
            distracted_count += 1

        details.append({
            "face":            face_idx + 1,
            "ear":             round(float(avg_ear), 3),
            "eyes_open":       bool(eyes_open),
            "facing_forward":  bool(facing_forward),
            "status":          "attentive" if is_attentive else "distracted",
        })

    total = attentive_count + distracted_count
    score = round((attentive_count / total) * 100, 1) if total > 0 else 0

    print(f"Faces: {total} | Attentive: {attentive_count} | Score: {score}%")
    return {
        "attentive":   attentive_count,
        "distracted":  distracted_count,
        "total_faces": total,
        "score":       score,
        "details":     details,
    }
