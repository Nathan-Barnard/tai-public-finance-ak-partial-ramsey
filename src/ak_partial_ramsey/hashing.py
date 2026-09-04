"""Canonical serialisation and hashing.

Configuration and parameter hashes appear in every diagnostic report. They must be
stable across processes and platforms, so floats are serialised through ``repr``, which
round-trips exactly in Python, rather than through a lossy fixed-precision format.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialise ``value`` deterministically: sorted keys, exact float repr, no spaces."""
    return json.dumps(
        _canonicalise(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def _canonicalise(value: Any) -> Any:
    if isinstance(value, float):
        # repr round-trips exactly for Python floats; format() and str() do not in general.
        return {"__float__": repr(value)}
    if isinstance(value, dict):
        return {str(k): _canonicalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalise(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "as_dict"):
        return _canonicalise(value.as_dict())
    return {"__repr__": repr(value)}


def digest_mapping(value: Any) -> str:
    """Return the SHA-256 hex digest of the canonical serialisation of ``value``."""
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()
