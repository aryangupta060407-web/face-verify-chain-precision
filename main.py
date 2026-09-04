"""Backward-compatible face -> web discovery -> blockchain verification CLI."""
import json
import sys

from face_id import get_face_encoding
from web_search import reverse_image_search
from blockchain import store_hash, verify_hash
from ai_evidence import explain_evidence


def run_pipeline(image_path: str, tolerance: float = 0.48):
    print("=== Step 1: Face detection & encoding ===")
    encoding = get_face_encoding(image_path)
    if encoding is None:
        print("No face detected — aborting.")
        return None
    print(f"Face detected and encoded. Dimension: {len(encoding)}")

    print("\n=== Step 2: Web discovery & face verification ===")
    print(f"Searching Google Lens exact matches, visual matches, and Yandex (threshold={tolerance})...")
    match = reverse_image_search(image_path, query_encoding=encoding, tolerance=tolerance)
    if match is None:
        print("No reliable public match found.")
        return None

    print(f"Google Lens exact matches: {match.get('google_exact_matches', 0)}")
    print(f"Google Lens visual matches: {match.get('google_visual_matches', 0)}")
    print(f"Context clues extracted: {len(match.get('context_clues', []))}")
    print(f"Targeted Google candidates: {match.get('targeted_google_candidates', 0)}")
    print(f"OpenAI web-search candidates: {match.get('openai_web_search_candidates', 0)}")
    print(f"Yandex matches: {match.get('yandex_matches', 0)}")
    print(f"Candidates checked: {match.get('candidates_checked', 0)}")
    print(f"Face-bearing candidates: {match.get('face_bearing_candidates', 0)}")
    print("\nBest match:")
    print(f"Title: {match.get('title', '')}")
    print(f"Source: {match.get('source', '')}")
    print(f"URL: {match.get('link', '')}")
    print(f"Engine: {match.get('engine', '')}")
    print(f"Face similarity: {match.get('face_similarity', 0.0):.4f}")
    print(f"Image match: {match.get('image_similarity', 0.0):.4f}")
    print(f"Image SHA-256: {match.get('image_sha256', '')}")
    evidence = explain_evidence(match.get("ranked_candidates", [match]))
    print(f"AI evidence layer: {'configured model' if evidence.get('model_used') else 'deterministic/local'}")
    print(f"Evidence summary: {evidence.get('summary', '')}")

    print("\n=== Step 3: Blockchain upload ===")
    record = dict(match)
    record["ai_evidence"] = evidence
    fingerprint_data = json.dumps(record, sort_keys=True).encode("utf-8")
    content_hash, tx_hash, status = store_hash(fingerprint_data)
    print(f"Hash: {content_hash.hex()}")
    print(f"Tx: {tx_hash} (status={status})")

    print("\n=== Step 4: Re-verification ===")
    result = verify_hash(fingerprint_data)
    print(f"On-chain record: {result}")
    verified = bool(result.get("exists"))
    print("VERIFIED" if verified else "VERIFICATION FAILED")
    return {"match": match, "evidence": evidence, "chain": result, "verified": verified}


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python main.py <image_path> [tolerance]")
        sys.exit(1)
    tolerance = float(sys.argv[2]) if len(sys.argv) == 3 else 0.48
    run_pipeline(sys.argv[1], tolerance=tolerance)
