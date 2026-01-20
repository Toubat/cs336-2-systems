"""Compression utilities for efficient profile data transfer from Modal."""

import base64
import gzip
import json
from dataclasses import asdict, dataclass


@dataclass
class BenchmarkResult:
    """Complete benchmark result with metadata and profile stats."""

    size: str
    num_params: int
    batch_size: int
    context_length: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    num_steps: int
    forward: dict
    backward: dict

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkResult":
        return cls(**data)


def encode_profile(data: dict | list[dict]) -> str:
    """
    Encode profile data to compressed base64 string for efficient Modal transfer.

    Args:
        data: Profile results dict or list of dicts

    Returns:
        Base64-encoded gzip-compressed JSON string
    """
    json_bytes = json.dumps(data, separators=(",", ":")).encode("utf-8")
    compressed = gzip.compress(json_bytes, compresslevel=9)
    return base64.b64encode(compressed).decode("ascii")


def decode_profile(encoded: str) -> dict | list[dict]:
    """
    Decode compressed base64 profile data.

    Args:
        encoded: Base64-encoded gzip-compressed JSON string

    Returns:
        Decoded profile results
    """
    compressed = base64.b64decode(encoded.encode("ascii"))
    json_bytes = gzip.decompress(compressed)
    return json.loads(json_bytes.decode("utf-8"))


def encode_results(results: list[dict]) -> str:
    """Encode multiple benchmark results for Modal transfer."""
    return encode_profile(results)


def decode_results(encoded: str) -> list[dict]:
    """Decode benchmark results from Modal."""
    result = decode_profile(encoded)
    return result if isinstance(result, list) else [result]
