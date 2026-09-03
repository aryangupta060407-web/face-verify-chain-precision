"""Optional OpenAI Responses API web-search discovery provider.

This module discovers cited public pages using image context and web search. It
never decides whether two faces are the same; local ArcFace remains authoritative.
"""
from __future__ import annotations

import base64
import os
import re
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


def _image_data_url(image_path: str) -> str:
    suffix = os.path.splitext(image_path)[1].lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    with open(image_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _annotation_values(response: Any) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    for output_item in getattr(response, "output", []) or []:
        content_items = getattr(output_item, "content", []) or []
        for content in content_items:
            annotations = getattr(content, "annotations", []) or []
            for annotation in annotations:
                if getattr(annotation, "type", "") != "url_citation":
                    continue
                url = getattr(annotation, "url", "") or ""
                if not url.startswith(("http://", "https://")):
                    continue
                found.append({
                    "url": url,
                    "title": getattr(annotation, "title", "") or "",
                })
    return found


def _page_image(url: str) -> str:
    """Read og:image/twitter:image from a cited public page when available."""
    try:
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "FaceVerifyChain/1.0 (evidence retrieval)"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for selector in (
            ('meta', {'property': 'og:image'}),
            ('meta', {'name': 'twitter:image'}),
            ('meta', {'property': 'twitter:image'}),
        ):
            tag = soup.find(*selector)
            if tag and tag.get("content", "").startswith(("http://", "https://")):
                return tag["content"]
    except requests.RequestException:
        pass
    return ""


def discover(image_path: str, max_results: int = 20) -> list[dict[str, Any]]:
    """Discover cited public pages; return normalized candidates for ArcFace."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=os.getenv("OPENAI_WEB_SEARCH_MODEL", "gpt-5.5"),
            tools=[{"type": "web_search", "search_context_size": "high"}],
            input=[{
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Use the public web search tool to find original public pages or posts "
                            "where this consented image, or a clearly matching public copy of the image, "
                            "appears. Use visible non-biometric context such as text, logos, usernames, "
                            "event names, or page metadata to form searches. Do not infer or state a "
                            "person's identity from facial appearance. Return only pages you actually "
                            "cite from web search. The local ArcFace verifier will independently check "
                            "candidate images later."
                        ),
                    },
                    {"type": "input_image", "image_url": _image_data_url(image_path)},
                ],
            }],
        )
    except Exception as exc:
        print(f"  [warn] OpenAI web discovery failed: {exc}")
        return []

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for citation in _annotation_values(response):
        url = citation["url"]
        if url in seen:
            continue
        seen.add(url)
        image_url = _page_image(url)
        parsed = urlparse(url)
        candidates.append({
            "title": citation["title"] or url,
            "source": parsed.netloc,
            "link": url,
            "thumbnail": image_url,
            "image": image_url,
            "engine": "OpenAI web search",
            "exact_matches": False,
            "openai_cited": True,
        })
        if len(candidates) >= max_results:
            break
    return candidates
