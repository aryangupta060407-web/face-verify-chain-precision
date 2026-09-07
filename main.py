"""Image -> public reverse-image discovery -> blockchain verification CLI."""
import json
import sys

from image_search import find_same_image
from blockchain import store_hash, verify_hash


def run_pipeline(image_path: str):
    print("=== Step 1: Image fingerprint ===")
    print("Creating an image-level fingerprint for the uploaded file...")

    print("\n=== Step 2: Public reverse-image discovery ===")
    match = find_same_image(image_path)
    if match is None:
        print("No same-image public post found.")
        return None

    print(f"Google Lens exact matches: {match.get('google_exact_matches', 0)}")
    print(f"Google Lens visual matches: {match.get('google_visual_matches', 0)}")
    print(f"Yandex matches: {match.get('yandex_matches', 0)}")
    print(f"Candidates discovered: {match.get('candidates_discovered', 0)}")
    print(f"Candidates checked: {match.get('candidates_checked', 0)}")
    print("\nBest matching post:")
    print(f"Title: {match.get('title', '')}")
    print(f"Source: {match.get('source', '')}")
    print(f"URL: {match.get('link', '')}")
    print(f"Engine: {match.get('engine', '')}")
    print(f"Image similarity: {match.get('image_similarity', 0.0):.4f}")
    print(f"Exact file match: {match.get('exact_file_match', False)}")
    print(f"Matched image SHA-256: {match.get('image_sha256', '')}")

    print("\n=== Step 3: Blockchain upload ===")
    fingerprint_data = json.dumps(match, sort_keys=True).encode("utf-8")
    content_hash, tx_hash, status = store_hash(fingerprint_data)
    print(f"Evidence hash: {content_hash.hex()}")
    print(f"Tx: {tx_hash} (status={status})")

    print("\n=== Step 4: Re-verification ===")
    result = verify_hash(fingerprint_data)
    print(f"On-chain record: {result}")
    verified = bool(result.get("exists"))
    print("VERIFIED" if verified else "VERIFICATION FAILED")
    return {"match": match, "chain": result, "verified": verified}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)
    run_pipeline(sys.argv[1])
