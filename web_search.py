"""Live reverse-image discovery with evidence-aware face verification."""
import hashlib
import io
import os
import requests
from PIL import Image
from serpapi import GoogleSearch
from dotenv import load_dotenv

from face_id import embeddings_from_bytes

load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")


def upload_temp_image(image_path: str) -> str:
    """Upload a consented query image to a temporary public URL for SerpApi."""
    with open(image_path, "rb") as f:
        r = requests.post("https://catbox.moe/user/api.php", data={"reqtype": "fileupload"}, files={"fileToUpload": f}, timeout=30)
    r.raise_for_status()
    url = r.text.strip()
    if not url.startswith("http"):
        raise RuntimeError(f"temporary image upload failed: {url}")
    return url


def _serpapi(params):
    params = dict(params)
    params["api_key"] = SERPAPI_KEY
    return GoogleSearch(params).get_dict()


def _normalise(item, engine, exact=False):
    return {
        "title": item.get("title", ""),
        "source": item.get("source", item.get("domain", "")),
        "link": item.get("link", item.get("url", "")),
        "thumbnail": item.get("thumbnail", ""),
        "image": item.get("image", item.get("original", item.get("image_url", ""))),
        "engine": engine,
        "exact_matches": bool(exact),
    }


def _google_lens(image_url):
    exact_response = _serpapi({"engine": "google_lens", "type": "exact_matches", "url": image_url})
    visual_response = _serpapi({"engine": "google_lens", "url": image_url})
    exact = [_normalise(x, "Google Lens", True) for x in (exact_response.get("exact_matches") or [])]
    visual = [_normalise(x, "Google Lens", False) for x in (visual_response.get("visual_matches") or [])]
    return exact, visual


def _yandex(image_url):
    response = _serpapi({"engine": "yandex_images", "url": image_url})
    raw = response.get("image_results") or response.get("inline_images") or response.get("results") or []
    return [_normalise(x, "Yandex", False) for x in raw]


def _download(url):
    if not url:
        return None
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "FaceVerifyChain/1.0"})
        r.raise_for_status()
        return r.content
    except requests.RequestException:
        return None


def _phash(data):
    try:
        img = Image.open(io.BytesIO(data)).convert("L").resize((32, 32))
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join("1" if p >= avg else "0" for p in pixels)
        return int(bits, 2)
    except Exception:
        return None


def _image_similarity(query_hash, candidate_hash):
    if query_hash is None or candidate_hash is None:
        return 0.0
    distance = (query_hash ^ candidate_hash).bit_count()
    return 1.0 - distance / 1024.0


def _face_metrics(query_encoding, data):
    faces = embeddings_from_bytes(data)
    if not faces:
        return None
    import numpy as np
    q = np.asarray(query_encoding, dtype=np.float32)
    q_norm = np.linalg.norm(q) or 1.0
    best = None
    for embedding, _box in faces:
        e = np.asarray(embedding, dtype=np.float32)
        cosine = float(np.dot(q, e) / (q_norm * (np.linalg.norm(e) or 1.0)))
        # ArcFace uses cosine similarity; the OpenCV fallback is deliberately
        # held to a stricter cosine threshold because it is not identity-grade.
        threshold = 0.45 if q.size >= 256 else 0.93
        if best is None or cosine > best["face_similarity"]:
            best = {"face_similarity": cosine, "faces_detected": len(faces), "face_threshold": threshold}
    return best


def reverse_image_search(image_path: str, query_encoding=None, max_candidates: int = 20, tolerance: float = 0.48):
    """Search exact, visual, and Yandex results, then rank verified candidates."""
    if not SERPAPI_KEY:
        raise RuntimeError("SERPAPI_KEY not set. Copy .env.example to .env and fill it in.")
    if query_encoding is None:
        raise ValueError("Face verification is required; unverified visual results are never returned.")

    image_url = upload_temp_image(image_path)
    exact, visual = _google_lens(image_url)
    try:
        yandex = _yandex(image_url)
    except Exception as exc:
        print(f"  [warn] Yandex search failed: {exc}")
        yandex = []

    all_candidates = []
    seen = set()
    for c in exact + visual + yandex:
        key = c.get("link") or c.get("image") or c.get("thumbnail")
        if key and key not in seen:
            seen.add(key)
            all_candidates.append(c)

    query_data = open(image_path, "rb").read()
    query_hash = _phash(query_data)
    ranked = []
    checked = 0
    face_bearing = 0
    for candidate in all_candidates[:max_candidates]:
        data = _download(candidate.get("image")) or _download(candidate.get("thumbnail"))
        if data is None:
            continue
        checked += 1
        metrics = _face_metrics(query_encoding, data)
        if metrics is None:
            continue
        face_bearing += 1
        image_match = _image_similarity(query_hash, _phash(data))
        face_similarity = metrics["face_similarity"]
        threshold = metrics["face_threshold"]
        candidate = dict(candidate)
        candidate.update(metrics)
        candidate["image_similarity"] = image_match
        candidate["image_sha256"] = hashlib.sha256(data).hexdigest()
        candidate["face_verified"] = face_similarity >= threshold
        candidate["reliable_match"] = candidate["face_verified"]
        # Exact results and strong face similarity dominate ordinary visuals.
        source_priority = 3 if candidate["exact_matches"] else (2 if candidate["engine"] == "Google Lens" else 1)
        candidate["ranking"] = (source_priority, face_similarity, image_match)
        ranked.append(candidate)

    ranked.sort(key=lambda x: x["ranking"], reverse=True)
    accepted = [x for x in ranked if x["reliable_match"]]
    if not accepted:
        print("No reliable public match found.")
        return None

    best = accepted[0]
    best.update({
        "google_exact_matches": len(exact),
        "google_visual_matches": len(visual),
        "yandex_matches": len(yandex),
        "candidates_checked": checked,
        "face_bearing_candidates": face_bearing,
        "image_sha256": best.get("image_sha256"),
    })
    print(f"  Best match ranked: {best.get('title', '?')[:80]}")
    return best


if __name__ == "__main__":
    import sys
    from face_id import get_face_encoding
    if len(sys.argv) != 2:
        print("Usage: python web_search.py <image_path>")
        raise SystemExit(1)
    enc = get_face_encoding(sys.argv[1])
    print(reverse_image_search(sys.argv[1], query_encoding=enc))
