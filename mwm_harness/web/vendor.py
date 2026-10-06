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


# The command-center look: three.js for the scene and the three fonts of the design.
# Each tarball is checked against the sha512 the npm registry published (read 2026-10-06);
# only the named members are kept, under the names on the right.
DESIGN_VERSION = "1"
DESIGN_PACKAGES = (
    (
        "https://registry.npmjs.org/three/-/three-0.149.0.tgz",
        "tohpUxPDht0qExRLDTM8sjRLc5d9STURNrdnK3w9A+V4pxaTBfKWWT/IqtiLfg23Vfc3Z+ImNfvRw1/0CtxrkQ==",
        {"package/build/three.min.js": "three.min.js", "package/LICENSE": "LICENSE-three"},
    ),
    (
        "https://registry.npmjs.org/@fontsource-variable/bricolage-grotesque/-/bricolage-grotesque-5.3.0.tgz",
        "TLi9Q4hJjS2UvoTMRSS2nHu6c4R56lAw60NR9QYtVRCHn0XtsFpiEhNffZ8Glsoxu6wEEwLKBP8lb94J52PNBA==",
        {
            "package/files/bricolage-grotesque-latin-wght-normal.woff2": "bricolage.woff2",
            "package/LICENSE": "LICENSE-bricolage",
        },
    ),
    (
        "https://registry.npmjs.org/@fontsource/jetbrains-mono/-/jetbrains-mono-5.3.0.tgz",
        "fqDfB5I9f1p1TV486aUgB9t8zP84P0O1FtQR5Ol9vjwPy+S+EIGlVYm1cvj2W5shcZMTg2nZFdVMoH5wFu8a1A==",
        {
            "package/files/jetbrains-mono-latin-400-normal.woff2": "jetbrains-400.woff2",
            "package/files/jetbrains-mono-latin-500-normal.woff2": "jetbrains-500.woff2",
            "package/files/jetbrains-mono-latin-600-normal.woff2": "jetbrains-600.woff2",
            "package/LICENSE": "LICENSE-jetbrains-mono",
        },
    ),
    (
        "https://registry.npmjs.org/@fontsource/newsreader/-/newsreader-5.3.0.tgz",
        "AFR6ZKY89GNL+9qBI2P2+66NhD5bhOj/v6fsLnFYLM4E0JnMB1yKb+Ro8cA64i1hwAWoBfTq5CHJGWH1toLDkw==",
        {
            "package/files/newsreader-latin-400-italic.woff2": "newsreader-400-italic.woff2",
            "package/files/newsreader-latin-500-italic.woff2": "newsreader-500-italic.woff2",
            "package/LICENSE": "LICENSE-newsreader",
        },
    ),
)


def design_dir() -> Path:
    return monaco_dir().parent / f"design-{DESIGN_VERSION}"


def design_ready(root: Path | None = None) -> bool:
    root = root or design_dir()
    return all((root / name).is_file() for *_, keep in DESIGN_PACKAGES for name in keep.values())


def unpack_members(blob: bytes, sha512: str, keep: dict[str, str], target: Path) -> int:
    """Verify ``blob``, then write each member named in ``keep`` to ``target`` under its new name."""
    digest = base64.b64encode(hashlib.sha512(blob).digest()).decode()
    if digest != sha512:
        raise VendorError(f"sha512 mismatch for {next(iter(keep.values()))}: nothing unpacked")
    found = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        for member in archive:
            name = keep.get(member.name)
            source = archive.extractfile(member) if name and member.isfile() else None
            if source is None:
                continue
            (target / name).write_bytes(source.read())
            found += 1
    if found != len(keep):
        raise VendorError(f"tarball misses files: found {found} of {len(keep)}")
    return found


def vendor_design(root: Path | None = None) -> tuple[Path, int]:
    root = root or design_dir()
    staging = root.with_name(root.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    count = 0
    try:
        for url, sha512, keep in DESIGN_PACKAGES:
            try:
                response = httpx.get(url, follow_redirects=True, timeout=120.0)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise VendorError(f"download failed: {exc}") from exc
            count += unpack_members(response.content, sha512, keep, staging)
    except VendorError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(root, ignore_errors=True)
    staging.rename(root)
    return root, count
