# Face Identification & Blockchain Verification Pipeline

Built for HH Goa 2026 Shortlisting — Task 3.

## What it does

This consent-based demo accepts a face photo, uploads it to a temporary public image host required by the search provider, performs genuine Google Lens exact-match and visual-match searches plus Yandex reverse-image search, downloads full-size candidate images when available, verifies every detected candidate face, ranks the evidence, and anchors the selected post metadata on a local blockchain.

```text
face scan → face embeddings → Lens exact/visual + Yandex discovery
          → full-image candidate verification → ranked match or no match
          → SHA-256 evidence hash → Ganache blockchain → independent verification
```

The system is designed to prefer **“No reliable public match found”** over presenting a visually similar stranger. Scores are face-similarity signals, not identity certainty, and images must be supplied or used with consent.

## Face embeddings and precision

When the optional ArcFace profile is installed, the project uses InsightFace/ONNX Runtime embeddings as its primary face representation. Without that optional profile, it uses the deterministic OpenCV fallback so the base installation remains practical on Windows and offline machines. The OpenCV Haar detector remains available as a fallback detector.

Every detected face in each candidate image is embedded and compared with the query face using cosine similarity. Full-size candidate URLs are preferred over thumbnails, and public X, Twitter, Instagram, and LinkedIn page candidates are checked before other domains when returned by the providers. Google Lens `exact_matches` are prioritized; strong Google Lens and Yandex face matches follow; ordinary visual results are ranked last. Exact candidates additionally receive an image-level perceptual-hash signal. Missing preview images can be recovered from public page metadata such as `og:image` or `twitter:image`.

Weak candidates are rejected. The application never returns an unverified first result and prints:

```text
No reliable public match found.
```

when no candidate passes the model-appropriate threshold.

## Optional AI evidence layer

The optional OpenAI discovery layer uses the Responses API `web_search` tool with the consented input image. It extracts only URLs actually cited by the response, reads public-page image metadata when available, and sends those candidate images through the existing local ArcFace verifier. The model is not permitted to identify a person or override biometric verification. The `ai_evidence.py` module then summarizes already-retrieved metadata; without `OPENAI_API_KEY`, the deterministic local summary is used.

The optional `FACE_SEARCH_PROVIDER_URL` and `FACE_SEARCH_API_KEY` settings support an authorized official face-search provider that returns a JSON `results`, `matches`, or `images` array. This is an adapter contract, not a built-in PimEyes/FaceCheck integration; no provider is hardcoded and no private service is scraped. Provider candidates still go through local face verification before they can be accepted.

## Blockchain used

The project uses local **Ganache** and the `HashRegistry` Solidity contract. It stores a SHA-256 fingerprint of the discovered post metadata, reads it back from the chain, and compares it independently. No public testnet is required. Ganache state resets when the local chain is restarted.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional ArcFace quality upgrade:
# pip install -r requirements-arcface.txt
npm --prefix frontend install
cp .env.example .env
```

Fill in `SERPAPI_KEY` in `.env` for Google Lens/Yandex search. Alternatively, configure `OPENAI_API_KEY` to enable Responses API web-search discovery, or configure both `FACE_SEARCH_PROVIDER_URL` and `FACE_SEARCH_API_KEY` for an authorized provider. `OPENAI_WEB_SEARCH_MODEL` defaults to `gpt-5.5`; `OPENAI_EVIDENCE_MODEL` defaults to `gpt-5-mini`. The SerpApi reverse-image path requires a public URL, so it uploads the consented query image to the configured temporary host before searching. OpenAI discovery uses the image as an input and keeps only web-cited public URLs. Start Ganache and deploy the contract:

```bash
ganache
python scripts/deploy_contract.py
```

Copy the printed contract address into `.env` as `CONTRACT_ADDRESS`.

## Running the pipeline

Standalone face check:

```bash
python face_id.py path/to/photo.jpg
```

Full backward-compatible CLI:

```bash
python main.py path/to/photo.jpg
```

An optional stricter tolerance can be provided as the second argument:

```bash
python main.py path/to/photo.jpg 0.45
```

The Flask API is available with:

```bash
python backend/app.py
```

The React frontend can be built with:

```bash
npm --prefix frontend run build
```

## CLI output

The full run reports OpenAI web-search candidate count, dedicated-provider count, Google Lens exact-match count, Google Lens visual-match count, Yandex count, candidate images checked, face-bearing candidates, selected title/source/URL/engine, face similarity, image similarity, SHA-256 hash, optional evidence summary, blockchain transaction, on-chain record, and final `VERIFIED` status.

## Limitations and privacy

This searches publicly indexed web content through SerpApi; it does **not** search the entire internet or private social-media databases. Search-engine recall for ordinary private individuals may be poor, and the correct outcome may be no reliable match. Reverse-image search results can be visually similar rather than the same person, which is why the face-verification and rejection layer is mandatory.

Face similarity is not proof of identity and should not be shown as “100% identity.” Lighting, pose, resolution, occlusion, model availability, and candidate-image quality affect scores. The query image is uploaded to a temporary public image host to enable remote reverse-image search; use only consented images and replace that uploader with an organization-controlled store for production use.
