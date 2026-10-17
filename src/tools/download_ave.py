"""Downloads AVE's labels and a seeded sample of its scenes into data/raw/ave/: python3 -m src.tools.download_ave [--count 200]"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "ave"

LABELS = (
    "https://drive.usercontent.google.com/download"
    "?id=1b_4yO94UbkkUAiRo4TB6QLfNef4WQ3t-&export=download&confirm=t"
)
YT_DLP = (
    "uv",
    "tool",
    "run",
    "--python",
    "3.12",
    "--from",
    "yt-dlp[default]",
    "--with",
    "deno",
    "yt-dlp",
)
FORMAT = "vcodec:h264,res:720"  # H.264 so OpenCV decodes it
SEED = 0
ATTEMPTS = 3
GONE = (
    "not available",
    "unavailable",
    "private video",
    "removed",
    "in your country",
    "confirm your age",
)


def labels() -> dict[str, Any]:
    path = OUT / "annotations.json"
    if not path.exists():
        print("downloading annotations.json")
        partial = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(LABELS, partial)
        partial.replace(path)
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return meta


def download(scene: str) -> str:
    """Empty on success, else yt-dlp's last error line."""
    result = subprocess.run(
        [
            *YT_DLP,
            "--quiet",
            "--no-warnings",
            "-S",
            FORMAT,
            "--merge-output-format",
            "mp4",
            "--sleep-requests",
            "1",
            "-P",
            str(OUT),
            "-o",
            "%(id)s.%(ext)s",
            f"https://youtu.be/{scene}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and (OUT / f"{scene}.mp4").exists():
        return ""
    lines = result.stderr.strip().splitlines()
    return lines[-1] if lines else f"yt-dlp exited {result.returncode}"


def gone(error: str) -> bool:
    text = error.lower()
    return "try again later" not in text and any(g in text for g in GONE)


def main() -> None:
    p = argparse.ArgumentParser(description="Download AVE's labels and scenes")
    p.add_argument("--count", type=int, default=200, help="scenes to have on disk")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    scenes = sorted(labels())
    random.Random(SEED).shuffle(scenes)

    skip_path = OUT / "unavailable.txt"
    unavailable: set[str] = (
        set(skip_path.read_text().split()) if skip_path.exists() else set()
    )
    have = {path.stem for path in OUT.glob("*.mp4")} & set(scenes)

    for scene in scenes:
        if len(have) >= args.count:
            break
        if scene in have or scene in unavailable:
            continue
        error = download(scene)
        for _ in range(ATTEMPTS - 1):
            if not error or gone(error):
                break
            time.sleep(10)
            error = download(scene)
        if not error:
            have.add(scene)
            print(f"[{len(have)}/{args.count}] {scene}")
            continue
        if not gone(error):
            sys.exit(f"{scene}: {error}\nstopped; rerun to resume")
        unavailable.add(scene)
        skip_path.write_text("\n".join(sorted(unavailable)) + "\n", encoding="utf-8")
        print(f"unavailable {scene}: {error}")

    print(f"{len(have)} scenes in {OUT}, {len(unavailable)} unavailable")


if __name__ == "__main__":
    main()
