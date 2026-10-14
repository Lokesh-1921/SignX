"""Display download requirements and, only with confirmation, stream a source URL.

Dataset access URLs and licences are intentionally supplied by the researcher;
the repository never embeds unverifiable URLs or silently downloads data.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
from urllib.request import Request, urlopen


def available_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def human_size(value: int | None) -> str:
    if value is None:
        return "unknown (check the official dataset release)"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return str(value)


def remote_size(url: str) -> int | None:
    try:
        with urlopen(Request(url, method="HEAD"), timeout=30) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value else None
    except OSError:
        return None


def stream_download(url: str, target: Path, checksum: str | None) -> None:
    temporary = target.with_suffix(target.suffix + ".part")
    start_at = temporary.stat().st_size if temporary.exists() else 0
    request = Request(url, headers={"Range": f"bytes={start_at}-"} if start_at else {})
    with urlopen(request, timeout=60) as response, temporary.open("ab" if start_at else "wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    if checksum:
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest.lower() != checksum.lower():
            raise RuntimeError("SHA-256 mismatch; preserved .part file for inspection/resume")
    os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset identifier used for local storage")
    parser.add_argument("--source-url", help="Official, licence-authorized archive URL")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--sha256", help="Official archive SHA-256, if published")
    parser.add_argument("--confirm", action="store_true", help="Authorize the actual transfer after reviewing output")
    arguments = parser.parse_args()
    target_dir = Path(arguments.output_dir) / arguments.dataset
    target_dir.mkdir(parents=True, exist_ok=True)
    size = remote_size(arguments.source_url) if arguments.source_url else None
    print(f"Dataset: {arguments.dataset}")
    print(f"Archive size: {human_size(size)}")
    print(f"Available disk: {human_size(available_bytes(target_dir))}")
    print(f"Destination: {target_dir.resolve()}")
    if not arguments.confirm:
        print("Planning only. Provide --source-url and --confirm after verifying licence, access rights, and disk use.")
        return
    if not arguments.source_url:
        raise SystemExit("--source-url is required with --confirm")
    if size is not None and size > available_bytes(target_dir):
        raise SystemExit("insufficient free disk space")
    filename = Path(arguments.source_url.split("?", 1)[0]).name or f"{arguments.dataset}.archive"
    target = target_dir / filename
    stream_download(arguments.source_url, target, arguments.sha256)
    print(f"Downloaded: {target}")


if __name__ == "__main__":
    main()
