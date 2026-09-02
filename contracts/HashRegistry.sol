// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title HashRegistry
/// @notice Stores a hash of discovered content (image/post fingerprint)
/// along with a timestamp, and lets anyone re-verify that a given hash
/// was recorded on-chain. Minimal by design for the hackathon deadline.
contract HashRegistry {
    struct Record {
        uint256 timestamp;
        address submitter;
        bool exists;
    }

    mapping(bytes32 => Record) public records;

    event HashStored(bytes32 indexed hash, address indexed submitter, uint256 timestamp);

    /// @notice Store a new hash on-chain. Reverts if already stored
    /// (so re-submission doesn't silently overwrite the original timestamp).
    function storeHash(bytes32 hash) external {
        require(!records[hash].exists, "Hash already recorded");
        records[hash] = Record(block.timestamp, msg.sender, true);
        emit HashStored(hash, msg.sender, block.timestamp);
    }

    /// @notice Check whether a hash exists on-chain and return its record.
    function verifyHash(bytes32 hash) external view returns (bool exists, uint256 timestamp, address submitter) {
        Record memory r = records[hash];
        return (r.exists, r.timestamp, r.submitter);
    }
}
