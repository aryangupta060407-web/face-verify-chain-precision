"""
CHECKPOINT 4: Full end-to-end pipeline.

    face scan -> reverse image/social search (with face verification)
              -> hash + store on-chain -> re-verify

Run:
    python main.py path/to/photo.jpg

This is the script you record your screen running for the submission.
"""
import sys
import json

from face_id import get_face_encoding
from web_search import reverse_image_search
from blockchain import store_hash, verify_hash


def run_pipeline(image_path: str, tolerance: float = 0.48):
    print("=== Step 1: Face detection & encoding ===")
    encoding = get_face_encoding(image_path)
    if encoding is None:
        print("No face detected — aborting.")
        return
    print(f"Face encoded ({len(encoding)}-d vector).\n")

    print("=== Step 2: Reverse image search with face verification ===")
    print(f"  Searching Google Lens + Yandex, then verifying each candidate")
    print(f"  thumbnail against the query face (tolerance={tolerance}; weak candidates are rejected)...\n")
    match = reverse_image_search(
        image_path, query_encoding=encoding, tolerance=tolerance
    )
    if match is None:
        print("No face-verified matching post found — aborting.")
        return
    print(f"\nFound VERIFIED match: {match['title']}")
    print(f"Source: {match['link']}")
    print(f"Engine: {match.get('engine', '?')}")
    print(f"Face verified: {match.get('face_verified', False)}\n")

    print("=== Step 3: Blockchain upload ===")
    # Fingerprint the discovered post's metadata (title + link + source).
    # Swap in the actual image/post bytes if you want to hash raw content instead.
    fingerprint_data = json.dumps(match, sort_keys=True).encode("utf-8")

    content_hash, tx_hash, status = store_hash(fingerprint_data)
    print(f"Hash: {content_hash.hex()}")
    print(f"Tx: {tx_hash} (status={status})\n")

    print("=== Step 4: Re-verification ===")
    result = verify_hash(fingerprint_data)
    print(f"On-chain record: {result}")

    if result["exists"]:
        print("\n✅ Verified: the discovered post's fingerprint is recorded on-chain.")
    else:
        print("\n❌ Verification failed.")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python main.py <image_path> [tolerance]")
        sys.exit(1)

    tol = float(sys.argv[2]) if len(sys.argv) == 3 else 0.48
    run_pipeline(sys.argv[1], tolerance=tol)
