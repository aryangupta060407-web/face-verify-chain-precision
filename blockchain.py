"""
CHECKPOINT 3b: Hash content, store on-chain, and re-verify.

Requires CONTRACT_ADDRESS in .env (from scripts/deploy_contract.py) and
contract_abi.json to exist alongside this file.

Run standalone to verify this step works before moving on:
    python blockchain.py "some post content or metadata string"

Prints the hash, the store tx, and then re-verifies it read back true.
If verify_hash returns exists=True with a matching timestamp, checkpoint
3b is done.
"""
import os
import sys
import json
import hashlib
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()

PROVIDER_URL = os.getenv("WEB3_PROVIDER_URL", "http://127.0.0.1:8545")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")
PRIVATE_KEY = os.getenv("DEPLOYER_PRIVATE_KEY")

ABI_PATH = os.path.join(os.path.dirname(__file__), "contract_abi.json")


def _load_contract(w3: Web3):
    with open(ABI_PATH) as f:
        abi = json.load(f)
    return w3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=abi)


def hash_content(data: bytes) -> bytes:
    """SHA-256 hash of arbitrary content (image bytes, JSON metadata, etc.)."""
    return hashlib.sha256(data).digest()


def store_hash(data: bytes):
    w3 = Web3(Web3.HTTPProvider(PROVIDER_URL))
    contract = _load_contract(w3)
    content_hash = hash_content(data)

    sender = w3.eth.account.from_key(PRIVATE_KEY).address if PRIVATE_KEY else w3.eth.accounts[0]

    tx = contract.functions.storeHash(content_hash).build_transaction({
        "from": sender,
        "nonce": w3.eth.get_transaction_count(sender),
        "gas": 200_000,
        "gasPrice": w3.to_wei("1", "gwei"),
    })

    if PRIVATE_KEY:
        signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    else:
        tx_hash = w3.eth.send_transaction(tx)

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return content_hash, tx_hash.hex(), receipt.status


def verify_hash(data: bytes):
    """Re-hashes `data` and checks whether that exact hash exists on-chain."""
    w3 = Web3(Web3.HTTPProvider(PROVIDER_URL))
    contract = _load_contract(w3)
    content_hash = hash_content(data)
    exists, timestamp, submitter = contract.functions.verifyHash(content_hash).call()
    return {"exists": exists, "timestamp": timestamp, "submitter": submitter}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python blockchain.py "content to hash and store"')
        sys.exit(1)

    content = sys.argv[1].encode("utf-8")

    content_hash, tx_hash, status = store_hash(content)
    print(f"Stored hash: {content_hash.hex()}")
    print(f"Tx: {tx_hash} (status={status})")

    result = verify_hash(content)
    print(f"Re-verification: {result}")
