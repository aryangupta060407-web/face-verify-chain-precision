"""Optional AI-assisted evidence normalization.

This module does not identify a person or make the biometric decision. It only
normalizes already-retrieved candidate metadata and explains source consistency.
ArcFace verification remains authoritative in web_search.py.
"""
from __future__ import annotations

import json
import os
from typing import Any


def normalize_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return deterministic, local normalization without an API call."""
    normalized = []
    for item in candidates:
        normalized.append({
            "url": item.get("link", ""),
            "title": item.get("title", ""),
            "platform": item.get("source", item.get("engine", "")),
            "image_url": item.get("image", item.get("thumbnail", "")),
            "engine": item.get("engine", ""),
            "provider_score": item.get("provider_score"),
            "face_similarity": item.get("face_similarity"),
            "image_similarity": item.get("image_similarity"),
            "face_verified": bool(item.get("face_verified", False)),
        })
    return normalized


def explain_evidence(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Optionally ask the configured OpenAI-compatible model for structured explanation.

    The model receives metadata only. It cannot override face_verified and is not
    used to infer identity from a face. If no key is configured, a deterministic
    explanation is returned.
    """
    normalized = normalize_candidates(candidates)
    verified = [x for x in normalized if x["face_verified"]]
    result = {
        "model_used": False,
        "verified_candidate_count": len(verified),
        "summary": (
            "At least one candidate passed the local face-verification threshold."
            if verified else
            "No candidate passed the local face-verification threshold."
        ),
        "candidates": normalized,
    }
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not normalized:
        return result

    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_EVIDENCE_MODEL", "gpt-5-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evidence-normalization assistant. Do not identify a person, "
                        "infer identity from facial appearance, or override local biometric fields. "
                        "Summarize only whether the retrieved URL, title, platform, and image metadata "
                        "are internally consistent. Output JSON only."
                    ),
                },
                {"role": "user", "content": json.dumps(normalized, ensure_ascii=False)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "evidence_summary",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "source_consistent": {"type": "boolean"},
                        },
                        "required": ["summary", "source_consistent"],
                        "additionalProperties": False,
                    },
                },
            },
            max_completion_tokens=400,
        )
        parsed = json.loads(response.choices[0].message.content)
        result.update(parsed)
        result["model_used"] = True
    except Exception as exc:
        result["model_error"] = str(exc)
    return result
