"""
CHECKPOINT 3a: Compile + deploy HashRegistry.sol.

Prereqs:
  1. `pip install py-solc-x web3`
  2. Ganache running locally (`npm install -g ganache && ganache`)
     OR a testnet RPC URL in .env (WEB3_PROVIDER_URL) + funded account.

Run:
    python scripts/deploy_contract.py

On success, prints the deployed contract address — copy it into your
.env as CONTRACT_ADDRESS. If this prints an address and a tx hash,
checkpoint 3a is done.
"""
import os
import json
from web3 import Web3
from solcx import compile_source, install_solc
from dotenv import load_dotenv

load_dotenv()

PROVIDER_URL = os.getenv("WEB3_PROVIDER_URL", "http://127.0.0.1:8545")
PRIVATE_KEY = os.getenv("DEPLOYER_PRIVATE_KEY")

CONTRACT_PATH = os.path.join(os.path.dirname(__file__), "..", "contracts", "HashRegistry.sol")


def compile_contract():
    install_solc("0.8.19")
    with open(CONTRACT_PATH, "r") as f:
        source = f.read()

    compiled = compile_source(source, output_values=["abi", "bin"], solc_version="0.8.19")
    contract_id, contract_interface = compiled.popitem()
    return contract_interface["abi"], contract_interface["bin"]


def deploy():
    w3 = Web3(Web3.HTTPProvider(PROVIDER_URL))
    assert w3.is_connected(), f"Could not connect to {PROVIDER_URL}"

    abi, bytecode = compile_contract()

    if PRIVATE_KEY:
        account = w3.eth.account.from_key(PRIVATE_KEY)
        sender = account.address
    else:
        # Ganache dev accounts are pre-funded and unlocked by default
        sender = w3.eth.accounts[0]

    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx = contract.constructor().build_transaction({
        "from": sender,
        "nonce": w3.eth.get_transaction_count(sender),
        "gas": 2_000_000,
        "gasPrice": w3.to_wei("1", "gwei"),
    })

    if PRIVATE_KEY:
        signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    else:
        tx_hash = w3.eth.send_transaction(tx)

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    print(f"Contract deployed at: {receipt.contractAddress}")
    print(f"Tx hash: {tx_hash.hex()}")

    # Save ABI for reuse by blockchain.py
    abi_path = os.path.join(os.path.dirname(__file__), "..", "contract_abi.json")
    with open(abi_path, "w") as f:
        json.dump(abi, f)
    print(f"ABI saved to {abi_path}")

    return receipt.contractAddress


if __name__ == "__main__":
    deploy()
