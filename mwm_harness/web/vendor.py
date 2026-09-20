"""Fetch the code viewer (Monaco) once, so the panel works with no network.

The npm tarball is checked against the sha512 the npm registry published for
this exact version (pinned below, read 2026-09-20) before anything is unpacked.
Only ``min/vs`` is kept. Nothing is committed: the files live in the user's data
folder and the panel serves them from there, with the CDN dropped from its
content security policy.
"""

from __future__ import annotations

import base64
import hashlib
import io
import shutil
import tarfile
from pathlib import Path

import httpx

MONACO_VERSION = "0.56.0"
TARBALL = f"https://registry.npmjs.org/monaco-editor/-/monaco-editor-{MONACO_VERSION}.tgz"
SHA512 = "sXboRm3BeBeLm938eaiyLMe0OxzfXIlZvbv4ir/jVgQy1zDhWjgmny0WoN45fuDKhCCQsYMbBJrv/A6jd8aCUg=="
PREFIX = "package/min/vs/"


class VendorError(Exception):
    pass


def monaco_dir() -> Path:
    share = Path.home() / ".local" / "share" / "mwm-harness" / "vendor"
    return share / f"monaco-{MONACO_VERSION}"


def monaco_ready(root: Path | None = None) -> bool:
    return ((root or monaco_dir()) / "vs" / "loader.js").is_file()


def unpack(blob: bytes, root: Path) -> int:
    """Verify, then extract ``min/vs`` into ``root/vs``. Returns the file count."""
    digest = base64.b64encode(hashlib.sha512(blob).digest()).decode()
    if digest != SHA512:
        raise VendorError(f"sha512 mismatch for monaco-editor {MONACO_VERSION}: nothing unpacked")
    staging = root.with_name(root.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    count = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith(PREFIX):
                continue
            relative = Path(member.name[len(PREFIX) :])
            if relative.is_absolute() or ".." in relative.parts:
                raise VendorError(f"unsafe path in the tarball: {member.name}")
            target = staging / "vs" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                continue
            target.write_bytes(source.read())
            count += 1
    if not (staging / "vs" / "loader.js").is_file():
        shutil.rmtree(staging, ignore_errors=True)
        raise VendorError("the tarball holds no min/vs/loader.js")
    shutil.rmtree(root, ignore_errors=True)
    staging.rename(root)
    return count


def vendor_monaco(root: Path | None = None) -> tuple[Path, int]:
    root = root or monaco_dir()
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = httpx.get(TARBALL, follow_redirects=True, timeout=120.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise VendorError(f"download failed: {exc}") from exc
    return root, unpack(response.content, root)
