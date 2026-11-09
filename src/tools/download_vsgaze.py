"""Downloads VSGaze's annotations into data/raw/vsgaze/ and writes its UCO-LAEO files as JSON; reading them needs
PyTables, which the image lacks, so run it through the image's uv:
docker compose run --rm -T --entrypoint uv pipeline run --no-project --python 3.10 --with pandas --with tables python src/tools/download_vsgaze.py"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "vsgaze"
URL = "https://zenodo.org/api/records/17357752/files/VSGaze.tar.gz/content"
SHA256 = "3bf42763c968b9684a3276e32b41403a43392282c67bdc8e461058ca949cb386"  # the .h5 columns hold pickles
SPLITS = ("train", "val", "test")
AGENT = {"User-Agent": "video-ai-thesis"}


def _plain(value: Any) -> Any:
    return value.tolist()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "VSGaze.tar.gz"
    if not archive.exists():
        partial = archive.with_name(archive.name + ".part")
        request = urllib.request.Request(URL, headers=AGENT)
        with urllib.request.urlopen(request) as response, partial.open("wb") as f:
            shutil.copyfileobj(response, f)
        partial.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        sys.exit(f"{archive} is not the pinned archive; delete it and run again")

    with tarfile.open(archive) as tar:
        for split in SPLITS:
            tar.extract(f"VSGaze/uco_laeo_{split}.h5", OUT, filter="data")
    for split in SPLITS:
        source = OUT / "VSGaze" / f"uco_laeo_{split}.h5"
        frame: Any = pd.read_hdf(source, "data")
        rows = frame.to_dict("records")
        source.with_suffix(".json").write_text(
            json.dumps(rows, default=_plain), encoding="utf-8"
        )
        print(f"{source.with_suffix('.json')}: {len(rows)} frames")


if __name__ == "__main__":
    main()
