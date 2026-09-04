"""Consent-based local face index for approved image corpora.

The index is deliberately local and transparent: it contains embeddings and
metadata for files the operator has explicitly placed in the corpus. It does
not crawl the internet or infer identity labels.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from face_id import get_face_embeddings

INDEX_VERSION = 1
DEFAULT_INDEX_DIR = "local_index"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _index_path(index_dir: str | os.PathLike[str]) -> Path:
    return Path(index_dir) / "index.json"


def build_index(corpus_dir: str, index_dir: str = DEFAULT_INDEX_DIR) -> dict[str, Any]:
    """Build an index from images explicitly placed in ``corpus_dir``."""
    root = Path(corpus_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Local index corpus not found: {root}")

    records: list[dict[str, Any]] = []
    images_seen = 0
    faces_indexed = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        images_seen += 1
        try:
            faces = get_face_embeddings(str(path))
        except Exception:
            faces = []
        for face_number, (embedding, _box) in enumerate(faces):
            vector = np.asarray(embedding, dtype=np.float32)
            norm = float(np.linalg.norm(vector)) or 1.0
            records.append({
                "path": str(path),
                "relative_path": str(path.relative_to(root)),
                "face_number": face_number,
                "embedding": (vector / norm).tolist(),
            })
            faces_indexed += 1

    output_dir = Path(index_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": INDEX_VERSION,
        "corpus_dir": str(root),
        "images_seen": images_seen,
        "faces_indexed": faces_indexed,
        "records": records,
    }
    _index_path(output_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_index(index_dir: str = DEFAULT_INDEX_DIR) -> dict[str, Any]:
    path = _index_path(index_dir).expanduser().resolve()
    if not path.is_file():
        return {"version": INDEX_VERSION, "records": [], "images_seen": 0, "faces_indexed": 0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != INDEX_VERSION:
        raise ValueError(f"Unsupported local index version: {payload.get('version')}")
    return payload


def search_index(query_encoding, index_dir: str = DEFAULT_INDEX_DIR, top_k: int = 20, threshold: float = 0.48) -> list[dict[str, Any]]:
    """Return top local records that pass the strict cosine threshold."""
    payload = load_index(index_dir)
    records = payload.get("records") or []
    if not records:
        return []
    query = np.asarray(query_encoding, dtype=np.float32)
    query /= float(np.linalg.norm(query) or 1.0)
    matches = []
    for record in records:
        vector = np.asarray(record.get("embedding", []), dtype=np.float32)
        if vector.size != query.size:
            continue
        similarity = float(np.dot(query, vector))
        if similarity >= threshold:
            matches.append({
                "title": record.get("relative_path", record.get("path", "Local image")),
                "source": "Local permitted face index",
                "link": "file://" + record.get("path", ""),
                "image": "",
                "thumbnail": "",
                "local_path": record.get("path", ""),
                "engine": "Local face index",
                "exact_matches": False,
                "face_similarity": similarity,
                "face_verified": True,
                "reliable_match": True,
                "indexed_face_number": record.get("face_number", 0),
            })
    matches.sort(key=lambda item: item["face_similarity"], reverse=True)
    return matches[:top_k]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build a local consented face index")
    parser.add_argument("corpus_dir")
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    args = parser.parse_args()
    result = build_index(args.corpus_dir, args.index_dir)
    print(f"Indexed {result['faces_indexed']} faces from {result['images_seen']} images")
