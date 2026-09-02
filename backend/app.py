"""
Flask API wrapping the face -> web search (with face verification) -> blockchain
pipeline so the React UI can call it directly instead of running the simulation.

Run:
    python backend/app.py

Requires everything from the main project's setup to already be done:
  - .env filled in (SERPAPI_KEY, CONTRACT_ADDRESS, etc.)
  - Ganache (or your chosen chain) running
  - Contract already deployed (scripts/deploy_contract.py)

Endpoints:
    POST /api/verify
    multipart/form-data with a single "image" file field.

    Response JSON:
    {
      "face": { "encodingLength": 128 },
      "match": { "title": "...", "link": "...", "source": "...", "engine": "...", "faceVerified": true },
      "chain": { "hash": "...", "tx": "...", "timestamp": 1234567890, "status": 1, "verified": true }
    }

    Query parameter (optional): ?tolerance=0.48  (default 0.48, lower = stricter)

    Error responses use {"error": "..."} with an appropriate HTTP status.
"""
import os
import sys
import json
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS

# Make the project root importable (face_id.py, web_search.py, blockchain.py
# live one directory up from backend/).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from face_id import get_face_encoding
from web_search import reverse_image_search
from blockchain import store_hash, verify_hash

app = Flask(__name__)
CORS(app)  # allow the Vite dev server (different port) to call this API


@app.route("/api/verify", methods=["POST"])
def verify():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided under the 'image' field."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    tolerance = float(request.args.get("tolerance", 0.48))

    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        encoding = get_face_encoding(tmp_path)
        if encoding is None:
            return jsonify({"error": "No face detected in the image."}), 422

        # Pass the query face encoding so web_search can verify each
        # candidate thumbnail is the SAME person before returning a match.
        match = reverse_image_search(
            tmp_path, query_encoding=encoding, tolerance=tolerance
        )
        if match is None:
            return jsonify({"error": "No face-verified matching post found for this face."}), 404

        fingerprint_data = json.dumps(match, sort_keys=True).encode("utf-8")
        content_hash, tx_hash, status = store_hash(fingerprint_data)
        result = verify_hash(fingerprint_data)

        return jsonify({
            "face": {"encodingLength": len(encoding)},
            "match": {
                "title": match["title"],
                "link": match["link"],
                "source": match.get("source"),
                "thumbnail": match.get("thumbnail"),
                "engine": match.get("engine"),
                "faceVerified": match.get("face_verified", False),
                "faceDistance": match.get("face_distance"),
                "similarity": match.get("similarity"),
                "reliableMatch": match.get("reliable_match", False),
                "candidatesChecked": match.get("candidates_checked"),
                "faceBearingCandidates": match.get("face_bearing_candidates"),
            },
            "chain": {
                "hash": content_hash.hex(),
                "tx": tx_hash,
                "status": status,
                "timestamp": result["timestamp"],
                "verified": result["exists"],
            },
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
