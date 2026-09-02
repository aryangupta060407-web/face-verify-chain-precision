# Face Identification & Blockchain Verification Pipeline

Built for HH Goa 2026 Shortlisting — Task 3.

## What it does

This consent-based demo accepts a face photo, performs genuine Google Lens and Yandex reverse-image searches through SerpApi, downloads candidate thumbnails, detects all faces in each candidate, and accepts a result only when the closest face passes a strict distance threshold. The accepted post metadata is then hashed with SHA-256, stored on a local Ethereum-compatible chain, read back, and independently verified.

```text
face scan → face encoding → live reverse-image search → candidate face verification
          → strict ranking/rejection → SHA-256 evidence hash → local blockchain → re-verification
```

The system is designed to prefer **“No face-verified matching post found”** over returning a visually similar stranger. Similarity is not proof of identity, and all test images must be supplied or used with consent.

## Precision safeguards

The default face distance tolerance is **0.48**, stricter than the common 0.60 default. The input stage selects the largest detected face. Candidate images are checked across all detected faces, ranked by their minimum face distance, and rejected when no candidate passes the threshold. The API and CLI expose the accepted face distance, similarity signal, candidate counts, and rejection message.

## Blockchain used

The project uses local **Ganache** by default. It stores a SHA-256 fingerprint of the discovered post metadata in the `HashRegistry` Solidity contract and reads it back for verification. Local chain state resets between sessions. No public testnet is required for the demo.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm install
cp .env.example .env
```

Fill in `SERPAPI_KEY` in `.env`, then start Ganache and deploy the contract:

```bash
ganache
python scripts/deploy_contract.py
```

Copy the printed contract address into `.env` as `CONTRACT_ADDRESS`.

## Running the pipeline

Checkpoint 1:

```bash
python face_id.py path/to/photo.jpg
```

Checkpoint 2:

```bash
python web_search.py path/to/photo.jpg
```

Full screen-recording path:

```bash
python main.py path/to/photo.jpg
```

An optional stricter threshold can be provided:

```bash
python main.py path/to/photo.jpg 0.45
```

The Flask API is available through:

```bash
python backend/app.py
```

The frontend can be started separately from `frontend/` with its package scripts.

## Known limitations

Reverse-image search depends on SerpApi and the search engines’ index. For ordinary private individuals, the actual web image may not be indexed, so the correct result is often no reliable match. The system does not search the entire internet by face and must not be used to identify unknown people without authorization.

Face distance is a model-dependent visual similarity measure, not a percentage certainty. Lighting, pose, resolution, occlusion, and multiple faces can affect it. SerpApi and image-hosting services require external network access and credentials. The local chain is a reproducible demo ledger, not a public blockchain record.
