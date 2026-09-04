"""Machine-readable diagnostic reports.

Every report carries the specification identity, the source-manifest hash, the
parameter and tolerance digests, the package and solver versions, wall time, and peak
memory. CS011 requires all of these on any run that informs a decision; recording them
by default means a report can never quietly lose its provenance.

Every report also carries its ``result_use_ceiling``, fixed at ``exploratory_only`` for
this repository's blocks.
"""

from __future__ import annotations

import json
import platform
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import BLOCKS_IMPLEMENTED, RESULT_USE_CEILING, SPEC_ID, SPEC_VERSION, __version__
from .hashing import digest_mapping

__all__ = [
    "Timer",
    "build_report",
    "environment_record",
    "peak_memory_bytes",
    "source_manifest_digest",
    "write_report",
]


def environment_record() -> dict[str, Any]:
    """Package and solver versions, interpreter, and platform."""
    import numpy
    import scipy

    return {
        "package": "ak-partial-ramsey",
        "package_version": __version__,
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }


def source_manifest_digest(root: Path | None = None) -> dict[str, Any]:
    """Hash ``docs/source_manifest.yaml`` so a report is tied to a provenance record.

    Returns ``available: False`` rather than raising when the manifest is absent - for
    instance when the package is imported from an installed wheel rather than the
    checkout. A report that says its provenance record is missing is more useful than
    one that fails to be written.
    """
    if root is None:
        root = Path(__file__).resolve().parents[2]
    path = root / "docs" / "source_manifest.yaml"
    if not path.is_file():
        return {"available": False, "path": str(path)}
    import hashlib

    data = path.read_bytes()
    return {
        "available": True,
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def peak_memory_bytes() -> int:
    """Peak resident set size of this process.

    ``ru_maxrss`` is in bytes on macOS and kilobytes on Linux; normalised here.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw) if sys.platform == "darwin" else int(raw) * 1024


@dataclass
class Timer:
    """Wall-clock timer for a report stage."""

    label: str
    _start: float = 0.0
    elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start


def build_report(
    *,
    stage: str,
    payload: dict[str, Any],
    parameters_digest: str | None = None,
    tolerances_digest: str | None = None,
    wall_seconds: float | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Assemble a complete diagnostic report around ``payload``."""
    report: dict[str, Any] = {
        "stage": stage,
        "specification": {
            "id": SPEC_ID,
            "version_implemented_against": SPEC_VERSION,
            "blocks_implemented": list(BLOCKS_IMPLEMENTED),
        },
        "result_use_ceiling": RESULT_USE_CEILING,
        "result_use_note": (
            "No output of this stage is an equilibrium result, a Ramsey optimum, a "
            "calibration, or a welfare result."
        ),
        "environment": environment_record(),
        "source_manifest": source_manifest_digest(root),
        "parameters_digest": parameters_digest,
        "tolerances_digest": tolerances_digest,
        "wall_seconds": wall_seconds,
        "peak_memory_bytes": peak_memory_bytes(),
        "payload": payload,
    }
    report["report_digest"] = digest_mapping(
        {k: v for k, v in report.items() if k != "report_digest"}
    )
    return report


def write_report(path: Path, report: dict[str, Any]) -> Path:
    """Write a report as indented JSON, refusing to overwrite existing evidence."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; solver logs never overwrite prior evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    return path
