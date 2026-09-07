"""Public reverse-image discovery for finding copies of the uploaded image.

This module deliberately matches the IMAGE, not the identity of a person in it.
It uses SerpApi/Google Lens for discovery and a local perceptual hash for
independent image-level verification.
"""
from __future__ import annotations

import hashlib
import io
import os
from urllib.parse import urljoin

import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

SOCIAL_DOMAINS = (
    "instagram.com", "x.com", "twitter.com", "facebook.com",
    "linkedin.com", "youtube.com", "pinterest.com", "reddit.com",
)


def _url_value(value):
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
        "engine": engine,
        "exact_matches": bool(exact),
    }


def _serpapi(params):
    if not SERPAPI_KEY:
        return {}
    from serpapi import GoogleSearch
    params = dict(params)
    params["api_key"] = SERPAPI_KEY
    return GoogleSearch(params).get_dict()


def _upload_image(image_path):
    """Create the public image reference required by image-search providers."""
    with open(image_path, "rb") as f:
        response = requests.post(
            "https://catbox.moe/user/api.php",
            data={"reqtype": "fileupload"},
            files={"fileToUpload": f},
            timeout=30,
        )
    response.raise_for_status()
    url = response.text.strip()
    if not url.startswith("http"):
        raise RuntimeError(f"temporary image upload failed: {url}")
    return url


def _page_image(page_url):
    if not page_url:
        return None
    try:
        response = requests.get(
            page_url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ImageVerifyChain/1.0)"},
        )
        response.raise_for_status()
        if (response.headers.get("content-type") or "").lower().startswith("image/"):
            return page_url
        soup = BeautifulSoup(response.text, "html.parser")
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
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ImageVerifyChain/1.0)"}
        if referer:
            headers["Referer"] = referer
        response = requests.get(url, timeout=25, headers=headers)
        response.raise_for_status()
        return response.content or None
    except requests.RequestException:
        return None


def _candidate_image(candidate):
    page_url = candidate.get("link", "")
    seen = set()
    for url in (_url_value(candidate.get("image")), _url_value(candidate.get("thumbnail"))):
        if not url or url in seen:
            continue
        seen.add(url)
        data = _download(url, referer=page_url)
        if data:
            return data
    page_image = _page_image(page_url)
    if page_image and page_image not in seen:
        return _download(page_image, referer=page_url)
    return None


def _phash(data):
    """64-bit DCT perceptual hash; tolerant of resize/compression/cropping noise."""
    try:
        image = Image.open(io.BytesIO(data)).convert("L").resize((32, 32))
        pixels = np.asarray(image, dtype=np.float32)
        # 2-D DCT without requiring scipy.
        n = 32
        x = np.arange(n, dtype=np.float32)
        basis = np.cos(np.pi * (2 * x[:, None] + 1) * np.arange(n)[None, :] / (2 * n))
        basis[:, 0] *= 1.0 / np.sqrt(2.0)
        dct = (2.0 / n) * basis.T @ pixels @ basis
        block = dct[:8, :8]
        median = np.median(block[1:, 1:])
        bits = block >= median
        return "".join("1" if bit else "0" for bit in bits.flat)
    except Exception:
        return None


def _phash_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    distance = sum(x != y for x, y in zip(a, b))
    return 1.0 - distance / len(a)


def _is_social(candidate):
    text = f"{candidate.get('link', '')} {candidate.get('source', '')}".lower()
    return any(domain in text for domain in SOCIAL_DOMAINS)


def find_same_image(image_path: str, max_candidates: int = 100):
    """Return the strongest public copy of the submitted image, if found."""
    if not SERPAPI_KEY:
        raise RuntimeError("SERPAPI_KEY is required for public reverse-image discovery.")

    query_data = open(image_path, "rb").read()
    query_sha256 = hashlib.sha256(query_data).hexdigest()
    query_hash = _phash(query_data)
    image_url = _upload_image(image_path)

    exact_response = _serpapi({"engine": "google_lens", "url": image_url, "type": "exact_matches"})
    visual_response = _serpapi({"engine": "google_lens", "url": image_url, "type": "visual_matches"})
    yandex_response = _serpapi({"engine": "yandex_images", "url": image_url})

    def records(response, *keys):
        result = []
        for key in keys:
            value = response.get(key) or []
            if isinstance(value, dict):
                value = value.get("results") or value.get("items") or value.get("matches") or []
            if isinstance(value, list):
                result.extend(x for x in value if isinstance(x, dict))
        return result

    candidates = (
        [_normalise(x, "Google Lens", True) for x in records(exact_response, "exact_matches", "exact_results", "image_sources")]
        + [_normalise(x, "Google Lens", False) for x in records(visual_response, "visual_matches", "visual_results")]
        + [_normalise(x, "Yandex", False) for x in records(yandex_response, "image_results", "inline_images", "results")]
    )

    seen = set()
    unique = []
    for candidate in candidates:
        key = candidate.get("link") or candidate.get("image") or candidate.get("thumbnail")
        if key and key not in seen:
            seen.add(key)
            unique.append(candidate)

    # Exact reverse-image results and public social pages get discovery priority,
    # but final acceptance is based on the downloaded image itself.
    unique.sort(key=lambda c: (bool(c.get("exact_matches")), _is_social(c)), reverse=True)

    ranked = []
    for candidate in unique[:max_candidates]:
        data = _candidate_image(candidate)
        if not data:
            continue
        candidate = dict(candidate)
        candidate_sha = hashlib.sha256(data).hexdigest()
        similarity = _phash_similarity(query_hash, _phash(data))
        candidate.update({
            "image_similarity": similarity,
            "exact_file_match": candidate_sha == query_sha256,
            "image_sha256": candidate_sha,
            "same_image": candidate_sha == query_sha256 or similarity >= 0.92,
            "social_source": _is_social(candidate),
        })
        ranked.append(candidate)

    ranked.sort(key=lambda c: (c["same_image"], c["exact_file_match"], c["image_similarity"], c["social_source"]), reverse=True)
    accepted = [c for c in ranked if c["same_image"]]
    if not accepted:
        print(f"No same-image match found. Candidates discovered: {len(unique)}, downloaded: {len(ranked)}")
        return None

    best = dict(accepted[0])
    best.update({
        "query_image_sha256": query_sha256,
        "google_exact_matches": len(records(exact_response, "exact_matches", "exact_results", "image_sources")),
        "google_visual_matches": len(records(visual_response, "visual_matches", "visual_results")),
        "yandex_matches": len(records(yandex_response, "image_results", "inline_images", "results")),
        "candidates_discovered": len(unique),
        "candidates_checked": len(ranked),
        "ranked_candidates": ranked[:10],
    })
    print(f"Same-image match: {best.get('title', '')[:100]}")
    print(f"Image similarity: {best['image_similarity']:.4f}")
    print(f"URL: {best.get('link', '')}")
    return best
