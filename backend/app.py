"""Flask API for image-level reverse-image discovery and blockchain verification."""
import os
import sys
import json
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from image_search import find_same_image
from blockchain import store_hash, verify_hash

app = Flask(__name__)
CORS(app)


@app.route("/api/verify", methods=["POST"])
def verify():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided under the 'image' field."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        match = find_same_image(tmp_path)
        if match is None:
            return jsonify({"error": "No matching public copy of this image was found."}), 404

        fingerprint_data = json.dumps(match, sort_keys=True).encode("utf-8")
        content_hash, tx_hash, status = store_hash(fingerprint_data)
        chain_result = verify_hash(fingerprint_data)

        return jsonify({
            "image": {"processed": True},
            "match": {
                "title": match["title"],
                "link": match["link"],
                "source": match.get("source"),
                "thumbnail": match.get("thumbnail"),
                "engine": match.get("engine"),
                "sameImage": match.get("same_image", False),
                "exactFileMatch": match.get("exact_file_match", False),
                "imageSimilarity": match.get("image_similarity", 0.0),
                "similarity": match.get("image_similarity", 0.0),
                "googleExactMatches": match.get("google_exact_matches", 0),
                "googleVisualMatches": match.get("google_visual_matches", 0),
                "yandexMatches": match.get("yandex_matches", 0),
                "candidatesDiscovered": match.get("candidates_discovered", 0),
                "candidatesChecked": match.get("candidates_checked", 0),
            },
            "chain": {
                "hash": content_hash.hex(),
                "tx": tx_hash,
                "status": status,
                "timestamp": chain_result["timestamp"],
                "verified": chain_result["exists"],
            },
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
