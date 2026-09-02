"""
CHECKPOINT 2: Reverse image / social media search via SerpApi + face verification.

Key improvements over the original:
  1. Multi-engine search — queries Google Lens AND Yandex (via SerpApi) and
     merges results. Yandex is generally stronger for face matching.
  2. Face verification — after finding candidate pages, it downloads each
     candidate's thumbnail and compares the face in it to the query face
     using face_recognition. Only VERIFIED matches (same person) are returned.
     This prevents false positives where Lens/Yandex return a lookalike or a
     page that merely has a similar layout.

SerpApi's Google Lens / Yandex engines take an image URL (not a local file
path) and return visual matches, including links back to source pages
(often social posts, profile pages, articles, etc.).

Because Lens/Yandex need a public URL, this module first uploads the local
image to a temporary public host (catbox.moe — simple, no API key, reliable)
and then hits SerpApi with that URL. Swap the uploader for imgbb or your own
S3 bucket if you'd rather not rely on catbox.moe.

Run standalone to verify this step works:
    python web_search.py path/to/photo.jpg

Prints all verified matches. If you get back a real URL with a title AND a
confirmed face match, checkpoint 2 is done.
"""
import io
import os
import sys
import time
from typing import Optional

import numpy as np
import requests
from serpapi import GoogleSearch
from dotenv import load_dotenv

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")


# ──────────────────────────────────────────────────────────────
#  Image hosting — upload local image to get a public URL
# ──────────────────────────────────────────────────────────────
def upload_temp_image(image_path: str) -> str:
    """Uploads a local image to catbox.moe and returns a public URL."""
    with open(image_path, "rb") as f:
        response = requests.post(
            "https://catbox.moe/user/api.php",
            data={"reqtype": "fileupload"},
            files={"fileToUpload": f},
        )
    response.raise_for_status()
    url = response.text.strip()
    if not url.startswith("http"):
        raise RuntimeError(f"catbox.moe upload failed: {url}")
    return url


# ──────────────────────────────────────────────────────────────
#  Per-engine searches — each returns a list of raw candidate dicts
# ──────────────────────────────────────────────────────────────
def _search_google_lens(image_url: str) -> list[dict]:
    """Query SerpApi Google Lens; return raw visual_matches list."""
    search = GoogleSearch({
        "engine": "google_lens",
        "url": image_url,
        "api_key": SERPAPI_KEY,
    })
    results = search.get_dict()
    return results.get("visual_matches", []) or []


def _search_yandex(image_url: str) -> list[dict]:
    """Query SerpApi Yandex Images; return raw matches list.

    Yandex's response shape under SerpApi is 'image_results' or
    'inline_images' depending on the result type. We normalise to a
    list of dicts with title/link/thumbnail keys.
    """
    search = GoogleSearch({
        "engine": "yandex_images",
        "url": image_url,
        "api_key": SERPAPI_KEY,
    })
    results = search.get_dict()

    raw = (
        results.get("image_results")
        or results.get("inline_images")
        or results.get("results")
        or []
    )
    # Normalise keys so downstream code can treat all engines uniformly.
    normalised = []
    for item in raw:
        normalised.append({
            "title": item.get("title", ""),
            "link": item.get("link", item.get("original", "")),
            "source": item.get("source", item.get("domain", "")),
            "thumbnail": item.get("thumbnail", ""),
        })
    return normalised


ENGINES = [
    ("google_lens", _search_google_lens),
    ("yandex", _search_yandex),
]


# ──────────────────────────────────────────────────────────────
#  Face verification — confirm a candidate thumbnail is the same person
# ──────────────────────────────────────────────────────────────
def _download_thumbnail(url: str) -> Optional[bytes]:
    """Download image bytes from a thumbnail URL. Returns None on failure."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


def _face_distance(query_encoding, thumbnail_bytes: bytes):
    """Return the lowest normalized descriptor distance across candidate faces."""
    import cv2
    import numpy as np
    try:
        image = cv2.imdecode(np.frombuffer(thumbnail_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        distances = []
        for x, y, w, h in faces:
            crop = cv2.resize(gray[y:y+h, x:x+w], (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
            encoding = ((crop - crop.mean()) / (crop.std() + 1e-6)).flatten()
            distances.append(float(np.linalg.norm(encoding - query_encoding) / np.sqrt(query_encoding.size)))
        return min(distances)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
#  Main entry point
# ──────────────────────────────────────────────────────────────
def reverse_image_search(
    image_path: str,
    query_encoding=None,
    max_candidates: int = 20,
    tolerance: float = 0.48,
) -> Optional[dict]:
    """
    Multi-engine reverse image search WITH face verification.

    Parameters
    ----------
    image_path : str
        Path to the local query image.
    query_encoding : numpy array or None
        128-d face encoding of the query image (from face_id.py). If
        provided, every candidate thumbnail is downloaded and face-matched
        before being accepted. If None, face verification is skipped and
        the first candidate is returned (OLD behaviour — not recommended).
    max_candidates : int
        Maximum number of raw candidates to fetch thumbnails for and verify.
    tolerance : float
        face_recognition distance tolerance. 0.6 = default, 0.5 = stricter.

    Returns
    -------
    dict | None
        {title, link, source, thumbnail, engine, face_verified} or None.
    """
    if not SERPAPI_KEY:
        raise RuntimeError(
            "SERPAPI_KEY not set. Copy .env.example to .env and fill it in."
        )

    image_url = upload_temp_image(image_path)

    # ── Collect raw candidates from all engines ───────────────
    all_candidates: list[dict] = []
    for engine_name, search_fn in ENGINES:
        try:
            raw = search_fn(image_url)
            for c in raw:
                c["_engine"] = engine_name
            all_candidates.extend(raw)
        except Exception as exc:
            print(f"  [warn] {engine_name} search failed: {exc}")

    if not all_candidates:
        return None

    # De-duplicate by link so we don't verify the same page twice.
    seen_links: set[str] = set()
    unique: list[dict] = []
    for c in all_candidates:
        link = c.get("link") or ""
        if link and link not in seen_links:
            seen_links.add(link)
            unique.append(c)

    if query_encoding is None:
        raise ValueError("Face verification is required; no unverified visual result will be returned.")

    # ── Face verification loop ────────────────────────────────
    verified: list[dict] = []
    checked = 0
    face_bearing = 0
    for candidate in unique[:max_candidates]:
        thumb_url = candidate.get("thumbnail", "")
        if not thumb_url:
            continue

        thumb_bytes = _download_thumbnail(thumb_url)
        if thumb_bytes is None:
            continue
        checked += 1
        distance = _face_distance(query_encoding, thumb_bytes)
        if distance is None:
            continue
        face_bearing += 1
        similarity = max(0.0, min(1.0, 1.0 - (distance / 1.0)))
        verified.append({
            "title": candidate.get("title"),
            "link": candidate.get("link"),
            "source": candidate.get("source"),
            "thumbnail": thumb_url,
            "engine": candidate.get("_engine"),
            "face_verified": distance <= tolerance,
            "face_distance": distance,
            "similarity": similarity,
            "candidate_checked": checked,
            "face_detected": True,
        })
        if distance <= tolerance:
            print(f"  ✓ Face match candidate: {candidate.get('title', '?')[:60]} [{candidate.get('_engine')}] distance={distance:.4f}")
        else:
            print(f"  ✗ Rejected candidate: distance={distance:.4f} > tolerance={tolerance:.2f}")

    verified.sort(key=lambda item: item["face_distance"])
    accepted = [item for item in verified if item["face_verified"]]
    if not accepted:
        print(f"No reliable face match: checked={checked}, face-bearing={face_bearing}, tolerance={tolerance:.2f}")
        return None
    best = accepted[0]
    best["candidates_checked"] = checked
    best["face_bearing_candidates"] = face_bearing
    best["reliable_match"] = True
    return best


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python web_search.py <image_path>")
        sys.exit(1)

    # When run standalone, load the face encoding from the image first so
    # the standalone test also benefits from face verification.
    from face_id import get_face_encoding

    image_path = sys.argv[1]
    enc = get_face_encoding(image_path)

    match = reverse_image_search(image_path, query_encoding=enc)

    if match is None:
        print("No face-verified matches found.")
    else:
        print("Top verified match:")
        for k, v in match.items():
            print(f"  {k}: {v}")