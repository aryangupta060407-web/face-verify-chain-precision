"""Live reverse-image discovery with evidence-aware face verification."""
import hashlib
import io
import os
import re
import requests
from PIL import Image
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib.parse import urljoin

from face_id import embeddings_from_bytes
from openai_discovery import discover as openai_web_discover, extract_context

load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
FACE_SEARCH_PROVIDER_URL = os.getenv("FACE_SEARCH_PROVIDER_URL")
FACE_SEARCH_API_KEY = os.getenv("FACE_SEARCH_API_KEY")


def upload_temp_image(image_path: str) -> str:
    """Upload a consented query image to a temporary public URL for SerpApi."""
    with open(image_path, "rb") as f:
        r = requests.post("https://catbox.moe/user/api.php", data={"reqtype": "fileupload"}, files={"fileToUpload": f}, timeout=30)
    r.raise_for_status()
    url = r.text.strip()
    if not url.startswith("http"):
        raise RuntimeError(f"temporary image upload failed: {url}")
    return url


def upload_lens_image_id(image_path: str) -> str:
    """Use SerpApi's native image upload when available, avoiding third-party URL rewriting."""
    with open(image_path, "rb") as image_file:
        response = requests.post(
            "https://serpapi.com/image",
            data={"api_key": SERPAPI_KEY},
            files={"image": (os.path.basename(image_path), image_file, "application/octet-stream")},
            timeout=45,
        )
    response.raise_for_status()
    payload = response.json()
    image_id = payload.get("image_id")
    if not image_id:
        raise RuntimeError(payload.get("error", "SerpApi image upload did not return image_id"))
    return str(image_id)


def _serpapi(params):
    try:
        from serpapi import GoogleSearch
    except ImportError as exc:
        raise RuntimeError("Install google-search-results to use SerpApi search.") from exc
    params = dict(params)
    params["api_key"] = SERPAPI_KEY
    return GoogleSearch(params).get_dict()


def _url_value(value):
    """Convert provider image/link fields, including nested objects, into a URL string."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("url", "link", "image", "original", "image_url", "thumbnail", "src"):
            result = _url_value(value.get(key))
            if result:
                return result
    return ""


def _normalise(item, engine, exact=False):
    return {
        "title": str(item.get("title", "") or ""),
        "source": str(item.get("source", item.get("domain", "")) or ""),
        "link": _url_value(item.get("link", item.get("url", ""))),
        "thumbnail": _url_value(item.get("thumbnail", "")),
        "image": _url_value(item.get("image", item.get("original", item.get("image_url", "")))),
        "snippet": str(item.get("snippet", item.get("description", "")) or ""),
        "displayed_link": str(item.get("displayed_link", "") or ""),
        "engine": engine,
        "exact_matches": bool(exact),
    }


def _lens_records(response, *keys):
    """Read Lens result arrays across SerpApi JSON layout variants."""
    records = []
    for key in keys:
        value = response.get(key) or []
        if isinstance(value, dict):
            value = value.get("results") or value.get("items") or value.get("matches") or []
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
    return records


def _google_lens(image_ref, use_image_id=False):
    request_base = {"engine": "google_lens"}
    request_base["image_id" if use_image_id else "url"] = image_ref
    exact_response = _serpapi({**request_base, "type": "exact_matches"})
    visual_response = _serpapi({**request_base, "type": "visual_matches"})
    exact_raw = _lens_records(exact_response, "exact_matches", "exact_results", "image_sources")
    visual_raw = _lens_records(visual_response, "visual_matches", "visual_results")
    exact = [_normalise(x, "Google Lens", True) for x in exact_raw]
    visual = [_normalise(x, "Google Lens", False) for x in visual_raw]
    return exact, visual


def _dedicated_provider(image_path: str):
    """Call an official configured provider; never scrape or guess a provider API."""
    if not FACE_SEARCH_PROVIDER_URL or not FACE_SEARCH_API_KEY:
        return []
    try:
        with open(image_path, "rb") as image_file:
            response = requests.post(
                FACE_SEARCH_PROVIDER_URL,
                headers={"Authorization": f"Bearer {FACE_SEARCH_API_KEY}"},
                files={"image": (os.path.basename(image_path), image_file, "application/octet-stream")},
                timeout=60,
            )
        response.raise_for_status()
        payload = response.json()
        raw = payload.get("results") or payload.get("matches") or payload.get("images") or []
        results = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            result = _normalise(item, "Dedicated provider", False)
            result["provider_score"] = item.get("score", item.get("similarity"))
            results.append(result)
        return results
    except (requests.RequestException, ValueError) as exc:
        print(f"  [warn] Dedicated provider failed: {exc}")
        return []


def _local_context_clues(image_path: str) -> list[str]:
    """Extract non-biometric clues already present in the submitted photo file."""
    clues = []
    try:
        import pytesseract
        with Image.open(image_path) as image:
            ocr_text = pytesseract.image_to_string(image)
        for line in re.split(r"[\n|]", ocr_text):
            text = re.sub(r"[^A-Za-z0-9@._ -]", " ", line).strip()
            if 3 <= len(text) <= 80 and text not in clues:
                clues.append(text)
    except (ImportError, OSError, ValueError):
        pass
    stem = os.path.splitext(os.path.basename(image_path))[0]
    stem = re.sub(r"[_-]+", " ", stem).strip()
    if stem and not re.fullmatch(r"(?:img|image|photo|dsc|pxl)?\s*\d+", stem, re.I):
        clues.append(stem)
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
            for tag_id, value in exif.items():
                if tag_id in (270, 315, 40092, 40093, 40094, 40095):
                    text = str(value).strip()
                    if 2 <= len(text) <= 120 and text not in clues:
                        clues.append(text)
    except (OSError, ValueError):
        pass
    return clues[:8]


def _google_context_search(clues: list[str], max_per_query: int = 5):
    """Search likely public profile pages using a bounded set of provenance-preserving queries."""
    results = []
    seen = set()
    sites = ("x.com", "instagram.com", "linkedin.com", "facebook.com", "pinterest.com")
    suffixes = ("", "profile")
    for clue in clues[:2]:
        safe_clue = " ".join(str(clue).split())[:100]
        for site in sites:
            for suffix in suffixes:
                query = f'site:{site} "{safe_clue}"' + (f" {suffix}" if suffix else "")
                response = _serpapi({"engine": "google", "q": query, "num": max_per_query})
                for item in response.get("organic_results") or []:
                    candidate = _normalise(item, f"Google Search ({site})", False)
                    candidate["context_clue"] = safe_clue
                    candidate["search_query"] = query
                    key = candidate.get("link") or candidate.get("title")
                    if key and key not in seen:
                        seen.add(key)
                        results.append(candidate)
    return results


def _yandex(image_url):
    response = _serpapi({"engine": "yandex_images", "url": image_url})
    raw = response.get("image_results") or response.get("inline_images") or response.get("results") or []
    return [_normalise(x, "Yandex", False) for x in raw]


def _priority_platform(candidate):
    """Prefer public social-platform pages during candidate verification."""
    text = f"{candidate.get('link', '')} {candidate.get('source', '')}".lower()
    for domain in ("x.com", "twitter.com", "instagram.com", "linkedin.com"):
        if domain in text:
            return 0
    return 1


def _page_image(page_url):
    """Resolve a public page to its declared preview image when search metadata lacks one."""
    if not page_url:
        return None
    try:
        r = requests.get(
            page_url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; FaceVerifyChain/1.0)"},
        )
        r.raise_for_status()
        content_type = (r.headers.get("content-type") or "").lower()
        if content_type.startswith("image/"):
            return page_url
        soup = BeautifulSoup(r.text, "html.parser")
        for attrs in (
            {"property": "og:image"},
            {"name": "twitter:image"},
            {"property": "twitter:image"},
        ):
            tag = soup.find("meta", attrs=attrs)
            value = tag.get("content", "") if tag else ""
            if value.startswith(("http://", "https://")):
                return value
            if value.startswith("/"):
                return urljoin(page_url, value)
    except (requests.RequestException, ValueError):
        return None
    return None


def _download(url, referer=None):
    if not url:
        return None
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; FaceVerifyChain/1.0)",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        }
        if referer:
            headers["Referer"] = referer
        r = requests.get(url, timeout=25, headers=headers)
        r.raise_for_status()
        if not r.content:
            return None
        return r.content
    except requests.RequestException:
        return None


def _candidate_image(candidate):
    """Try direct image fields first, then recover social-page preview metadata."""
    page_url = _url_value(candidate.get("link"))
    urls = [_url_value(candidate.get("image")), _url_value(candidate.get("thumbnail"))]
    seen = set()
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        data = _download(url, referer=page_url)
        if data is not None:
            return data
    page_image = _page_image(page_url)
    if page_image and page_image not in seen:
        return _download(page_image, referer=page_url)
    return None


def _candidate_order_key(candidate):
    """Order mixed provider results for verification without changing acceptance rules.

    This is deliberately a soft ordering heuristic. It uses provider metadata,
    direct-image availability, and page context to avoid spending the finite
    candidate budget on obvious shopping/catalog pages, but never rejects a
    candidate before the image is checked by ArcFace.
    """
    text = " ".join(str(candidate.get(field, "") or "") for field in ("title", "source", "link", "snippet", "context_clue")).lower()
    social = _priority_platform(candidate) == 0
    direct_image = bool(_url_value(candidate.get("image")) or _url_value(candidate.get("thumbnail")))
    human_terms = ("profile", "portrait", "person", "people", "face", "photo", "post", "avatar", "creator", "interview", "news", "team", "student")
    product_terms = ("product", "shop", "store", "buy", "price", "catalog", "clothing", "apparel", "fashion", "shoes", "dress", "collection")
    human_context = sum(term in text for term in human_terms)
    product_context = sum(term in text for term in product_terms)
    exact = bool(candidate.get("exact_matches"))
    return (social, exact, direct_image, human_context, -product_context)


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


def _face_metrics(query_encoding, data, tolerance=0.48):
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
        # Respect the caller's strict ArcFace threshold; the fallback metric
        # remains deliberately conservative because it is not identity-grade.
        threshold = 0.48 if q.size >= 256 else 0.93
        if best is None or cosine > best["face_similarity"]:
            best = {"face_similarity": cosine, "faces_detected": len(faces), "face_threshold": threshold}
    return best


def reverse_image_search(image_path: str, query_encoding=None, max_candidates: int = 100, tolerance: float = 0.48):
    """Search exact, visual, and Yandex results, then rank verified candidates."""
    if query_encoding is None:
        raise ValueError("Face verification is required; unverified visual results are never returned.")
    if not SERPAPI_KEY and not (FACE_SEARCH_PROVIDER_URL and FACE_SEARCH_API_KEY) and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Configure SERPAPI_KEY, OPENAI_API_KEY, or both FACE_SEARCH_PROVIDER_URL and FACE_SEARCH_API_KEY in .env.")

    context_clues = extract_context(image_path)
    for clue in _local_context_clues(image_path):
        if clue not in context_clues:
            context_clues.append(clue)
    openai_candidates = openai_web_discover(image_path)
    provider = _dedicated_provider(image_path)
    context_candidates = []
    exact, visual, yandex = [], [], []
    if SERPAPI_KEY:
        if context_clues:
            try:
                context_candidates = _google_context_search(context_clues)
            except Exception as exc:
                print(f"  [warn] Context Google search failed: {exc}")
        image_url = None
        try:
            image_id = upload_lens_image_id(image_path)
            exact, visual = _google_lens(image_id, use_image_id=True)
        except Exception as native_upload_error:
            print(f"  [warn] Native Lens upload failed; using public URL fallback: {native_upload_error}")
            image_url = upload_temp_image(image_path)
            exact, visual = _google_lens(image_url)
        if image_url is None:
            # Preserve the existing Yandex input path; only Lens uses image_id.
            image_url = upload_temp_image(image_path)
        try:
            yandex = _yandex(image_url)
        except Exception as exc:
            print(f"  [warn] Yandex search failed: {exc}")

    all_candidates = []
    seen = set()
    for c in openai_candidates + context_candidates + provider + exact + visual + yandex:
        key = _url_value(c.get("link")) or _url_value(c.get("image")) or _url_value(c.get("thumbnail"))
        if key and key not in seen:
            seen.add(key)
            all_candidates.append(c)

    print(f"Context clues extracted: {len(context_clues)}")
    print(f"Targeted Google candidates: {len(context_candidates)}")
    print(f"OpenAI web-search candidates: {len(openai_candidates)}")
    print(f"Dedicated provider matches: {len(provider)}")
    print(f"Google Lens exact matches: {len(exact)}")
    print(f"Google Lens visual matches: {len(visual)}")
    print(f"Yandex matches: {len(yandex)}")
    # Search providers return mixed results. Check public social-platform pages
    # first, while retaining exact-match priority within each group.
    all_candidates.sort(key=_candidate_order_key, reverse=True)
    print(f"Priority social candidates: {sum(_priority_platform(c) == 0 for c in all_candidates)}")
    print(f"Unique candidates discovered: {len(all_candidates)}")
    query_data = open(image_path, "rb").read()
    query_hash = _phash(query_data)
    ranked = []
    checked = 0
    face_bearing = 0
    download_failed = 0
    rejected = 0
    for candidate in all_candidates[:max_candidates]:
        data = _candidate_image(candidate)
        if data is None:
            download_failed += 1
            continue
        checked += 1
        metrics = _face_metrics(query_encoding, data, tolerance=tolerance)
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
        candidate["face_verified"] = face_similarity >= max(threshold, tolerance)
        candidate["reliable_match"] = candidate["face_verified"]
        if not candidate["reliable_match"]:
            rejected += 1
        # Face similarity is authoritative. Source/platform priority only
        # breaks ties and must never make a weaker lookalike outrank a stronger
        # biometric match from another public source.
        source_priority = 5 if candidate["engine"] == "OpenAI web search" else (4 if candidate["engine"].startswith("Google Search") else (4 if candidate["engine"] == "Dedicated provider" else (3 if candidate["exact_matches"] else (2 if candidate["engine"] == "Google Lens" else 1))))
        candidate["ranking"] = (face_similarity, source_priority, image_match)
        ranked.append(candidate)

    ranked.sort(key=lambda x: x["ranking"], reverse=True)
    if ranked:
        print("Top candidate face scores:")
        for item in ranked[:5]:
            print(f"  {item.get('face_similarity', 0.0):.4f} face / {item.get('image_similarity', 0.0):.4f} image | {item.get('engine')} | {item.get('link', '')[:100]}")
    accepted = [x for x in ranked if x["reliable_match"]]
    if not accepted:
        print(f"Candidates checked: {checked}; face-bearing: {face_bearing}; downloads failed: {download_failed}; rejected by face threshold: {rejected}")
        print("No reliable public match found.")
        return None

    best = accepted[0]
    best.update({
        "context_clues": context_clues,
        "targeted_google_candidates": len(context_candidates),
        "openai_web_search_candidates": len(openai_candidates),
        "dedicated_provider_matches": len(provider),
        "google_exact_matches": len(exact),
        "google_visual_matches": len(visual),
        "yandex_matches": len(yandex),
        "candidates_checked": checked,
        "face_bearing_candidates": face_bearing,
        "image_sha256": best.get("image_sha256"),
        "ranked_candidates": ranked[:10],
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
