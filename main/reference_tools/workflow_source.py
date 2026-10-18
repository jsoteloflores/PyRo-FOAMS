"""Pinned original-source text helpers for workflow replay tooling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union


class WorkflowSourceValidationError(ValueError):
    """Raised when original source bytes or extracted lines differ from the map."""

    def __init__(self, message: str, *, code: str, context: Optional[dict] = None) -> None:
        super().__init__(message)
        self.code = code
        self.context = dict(context or {})


@dataclass(frozen=True)
class SourceFragment:
    start_line: int
    end_line: int
    text: str
    raw_sha256: str
    lf_sha256: str


def _lf_bytes(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def source_hashes(path: Union[Path, str]) -> Tuple[str, str]:
    """Return raw archive hash and LF-normalized content hash, in that order."""

    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest(), hashlib.sha256(_lf_bytes(data)).hexdigest()


def extract_source_fragment(
    path: Union[Path, str],
    start_line: int,
    end_line: int,
    *,
    expected_lf_sha256: Optional[str] = None,
) -> SourceFragment:
    """Extract inclusive 1-based source lines after canonical LF normalization."""

    source_path = Path(path)
    if (
        not isinstance(start_line, int)
        or isinstance(start_line, bool)
        or not isinstance(end_line, int)
        or isinstance(end_line, bool)
        or start_line < 1
        or end_line < start_line
    ):
        raise WorkflowSourceValidationError(
            "source fragment line bounds are invalid",
            code="invalid_fragment_bounds",
            context={"start_line": start_line, "end_line": end_line},
        )

    raw = source_path.read_bytes()
    normalized = _lf_bytes(raw)
    lines = normalized.splitlines(keepends=True)
    if end_line > len(lines):
        raise WorkflowSourceValidationError(
            "source fragment extends beyond the pinned file",
            code="fragment_out_of_range",
            context={"path": str(source_path), "line_count": len(lines), "end_line": end_line},
        )
    fragment_bytes = b"".join(lines[start_line - 1 : end_line])
    fragment_hash = hashlib.sha256(fragment_bytes).hexdigest()
    if expected_lf_sha256 is not None and fragment_hash != expected_lf_sha256:
        raise WorkflowSourceValidationError(
            "source fragment hash does not match the extraction map",
            code="fragment_hash_mismatch",
            context={
                "path": str(source_path),
                "start_line": start_line,
                "end_line": end_line,
                "expected_lf_sha256": expected_lf_sha256,
                "actual_lf_sha256": fragment_hash,
            },
        )
    return SourceFragment(
        start_line=start_line,
        end_line=end_line,
        text=fragment_bytes.decode("utf-8"),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        lf_sha256=fragment_hash,
    )
