"""Face detection and normalized visual descriptor utilities."""
import io
import os
import sys
from typing import Optional

import cv2
import numpy as np
import requests
from PIL import Image

DETECTOR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def _faces_and_gray(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = DETECTOR.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    return gray, faces


def _largest(faces):
    return max(faces, key=lambda b: int(b[2]) * int(b[3])) if len(faces) else None


def _encode(gray: np.ndarray, box) -> np.ndarray:
    x, y, w, h = map(int, box)
    crop = cv2.resize(gray[y:y+h, x:x+w], (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    return ((crop - crop.mean()) / (crop.std() + 1e-6)).flatten()


def _load_image(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        image = cv2.imdecode(np.frombuffer(path_or_bytes, np.uint8), cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(path_or_bytes)
    return image


def get_face_encoding(image_path: str):
    image = _load_image(image_path)
    if image is None:
        return None
    gray, faces = _faces_and_gray(image)
    box = _largest(faces)
    return _encode(gray, box) if box is not None else None


def get_face_location(image_path: str):
    image = _load_image(image_path)
    if image is None:
        return None
    _, faces = _faces_and_gray(image)
    box = _largest(faces)
    if box is None:
        return None
    x, y, w, h = map(int, box)
    return y, x + w, y + h, x


def crop_face(image_path: str, output_path: str = None, margin: float = 0.3):
    location = get_face_location(image_path)
    if location is None:
        return None
    top, right, bottom, left = location
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    face_h, face_w = bottom - top, right - left
    top = max(0, top - int(face_h * margin)); bottom = min(height, bottom + int(face_h * margin))
    left = max(0, left - int(face_w * margin)); right = min(width, right + int(face_w * margin))
    cropped = img.crop((left, top, right, bottom))
    if output_path is None:
        base, _ = os.path.splitext(image_path)
        output_path = f"{base}_face_crop.jpg"
    cropped.save(output_path, "JPEG", quality=95)
    return output_path


def encoding_from_url(image_url: str):
    try:
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
        image = _load_image(resp.content)
        if image is None:
            return None
        gray, faces = _faces_and_gray(image)
        box = _largest(faces)
        return _encode(gray, box) if box is not None else None
    except Exception:
        return None


def find_best_match(original_encoding, candidates, tolerance: float = 0.48):
    ranked = []
    for candidate in candidates:
        thumb = candidate.get("thumbnail")
        if not thumb:
            continue
        candidate_encoding = encoding_from_url(thumb)
        if candidate_encoding is None:
            continue
        distance = float(np.linalg.norm(candidate_encoding - original_encoding) / np.sqrt(original_encoding.size))
        candidate = dict(candidate)
        candidate["face_distance"] = distance
        candidate["face_verified"] = distance <= tolerance
        ranked.append(candidate)
    accepted = sorted((c for c in ranked if c["face_verified"]), key=lambda c: c["face_distance"])
    return (accepted[0], accepted[0]["face_distance"]) if accepted else (None, None)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python face_id.py <image_path>")
        sys.exit(1)
    encoding = get_face_encoding(sys.argv[1])
    if encoding is None:
        print("No face detected in the image.")
    else:
        print(f"Face detected. Descriptor vector length: {len(encoding)}")
        print(f"Cropped face saved to: {crop_face(sys.argv[1])}")
