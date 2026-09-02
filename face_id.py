"""Consent-based face detection and embedding utilities.

InsightFace/ArcFace is used when installed and its model is available. The
OpenCV descriptor remains a deterministic fallback for offline demos.
"""
import io
import os
import sys
from typing import Optional

import cv2
import numpy as np
import requests
from PIL import Image

DETECTOR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_ARCFACE = None
_ARCFACE_TRIED = False


def _arcface_app():
    global _ARCFACE, _ARCFACE_TRIED
    if _ARCFACE_TRIED:
        return _ARCFACE
    _ARCFACE_TRIED = True
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _ARCFACE = app
    except Exception as exc:
        print(f"  [info] ArcFace unavailable; using OpenCV fallback ({exc.__class__.__name__})")
    return _ARCFACE


def _load_image(value):
    if isinstance(value, (bytes, bytearray)):
        return cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR)
    return cv2.imread(str(value))


def _opencv_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = DETECTOR.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    return gray, boxes


def _opencv_embedding(gray, box):
    x, y, w, h = map(int, box)
    crop = cv2.resize(gray[y:y+h, x:x+w], (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    return ((crop - crop.mean()) / (crop.std() + 1e-6)).flatten()


def get_face_embeddings(value):
    """Return embeddings and boxes for every face in an image."""
    image = _load_image(value)
    if image is None:
        return []

    app = _arcface_app()
    if app is not None:
        try:
            faces = app.get(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return [(np.asarray(face.embedding, dtype=np.float32), tuple(map(int, face.bbox))) for face in faces if face.embedding is not None]
        except Exception:
            pass

    gray, boxes = _opencv_faces(image)
    return [(_opencv_embedding(gray, box), tuple(map(int, box))) for box in boxes]


def get_face_encoding(image_path: str):
    faces = get_face_embeddings(image_path)
    if not faces:
        return None
    # Use the largest detected face as the query subject.
    return max(faces, key=lambda item: max(0, item[1][2] - item[1][0]) * max(0, item[1][3] - item[1][1]))[0]


def get_face_location(image_path: str):
    faces = get_face_embeddings(image_path)
    if not faces:
        return None
    box = max(faces, key=lambda item: max(0, item[1][2] - item[1][0]) * max(0, item[1][3] - item[1][1]))[1]
    if len(box) == 4 and box[2] > box[0] and box[3] > box[1]:
        # ArcFace boxes are x1,y1,x2,y2; OpenCV fallback boxes are x,y,w,h.
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            x2, y2 = x1 + x2, y1 + y2
        return y1, x2, y2, x1
    return None


def crop_face(image_path: str, output_path: str = None, margin: float = 0.3):
    location = get_face_location(image_path)
    if location is None:
        return None
    top, right, bottom, left = location
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    fh, fw = bottom - top, right - left
    top = max(0, top - int(fh * margin)); bottom = min(height, bottom + int(fh * margin))
    left = max(0, left - int(fw * margin)); right = min(width, right + int(fw * margin))
    if output_path is None:
        base, _ = os.path.splitext(image_path)
        output_path = f"{base}_face_crop.jpg"
    img.crop((left, top, right, bottom)).save(output_path, "JPEG", quality=95)
    return output_path


def embeddings_from_bytes(data: bytes):
    return get_face_embeddings(data)


def encoding_from_url(image_url: str):
    try:
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        faces = embeddings_from_bytes(resp.content)
        return faces[0][0] if faces else None
    except Exception:
        return None


def find_best_match(original_encoding, candidates, tolerance: float = 0.48):
    ranked = []
    for candidate in candidates:
        thumb = candidate.get("image") or candidate.get("thumbnail")
        if not thumb:
            continue
        enc = encoding_from_url(thumb)
        if enc is None:
            continue
        distance = float(np.linalg.norm(enc - original_encoding) / np.sqrt(original_encoding.size))
        if distance <= tolerance:
            item = dict(candidate); item["face_distance"] = distance; ranked.append(item)
    return (min(ranked, key=lambda x: x["face_distance"]), min(ranked, key=lambda x: x["face_distance"])["face_distance"]) if ranked else (None, None)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python face_id.py <image_path>")
        sys.exit(1)
    encoding = get_face_encoding(sys.argv[1])
    if encoding is None:
        print("No face detected in the image.")
    else:
        print(f"Face detected and encoded. Descriptor dimension: {len(encoding)}")
        print(f"Cropped face saved to: {crop_face(sys.argv[1])}")
