"""Downloads and unpacks VPCD into data/raw/vpcd/: python3 -m src.tools.download_vpcd"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "vpcd"
URL = "https://www.robots.ox.ac.uk/~abrown/VPCD.tar.gz"
SIZE = 5_690_329_235
CHUNK = 1 << 22


def main() -> None:
    archive = OUT / "VPCD.tar.gz"
    OUT.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or archive.stat().st_size != SIZE:
        partial = archive.with_name(archive.name + ".part")
        done = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(
            URL, headers={"User-Agent": "video-ai-thesis", "Range": f"bytes={done}-"}
        )
        with urllib.request.urlopen(request) as response, partial.open("ab") as f:
            while chunk := response.read(CHUNK):
                f.write(chunk)
        if partial.stat().st_size != SIZE:
            raise SystemExit(
                f"{partial} is {partial.stat().st_size} bytes, expected {SIZE}: run again to resume"
            )
        partial.rename(archive)
    if not (OUT / "VPCD_Release1" / "VPCD_README.txt").exists():
        with tarfile.open(archive) as tar:
            tar.extractall(OUT, filter="data")
    print(sorted(p.name for p in (OUT / "VPCD_Release1").iterdir()))


if __name__ == "__main__":
    main()
