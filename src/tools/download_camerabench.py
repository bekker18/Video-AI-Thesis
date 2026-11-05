"""Downloads CameraBench's test clips, labels, yes/no task files and official scorer into data/raw/camerabench/:
python3 -m src.tools.download_camerabench"""

from __future__ import annotations

import json
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "camerabench"

# Pinned, so a rerun fetches the same files
VIDEOS = ("syCen/Videos4CameraBench", "1ec8d55f8c380abbb6ea7087e64778f0ca76a68f")
LABELS = ("syCen/CameraBench", "659f73c8899d81ffebd85876e3cdd137435b5e1b")
CODE = ("linzhiqiu/t2v_metrics", "4c9a00e54ddad6f7ba1199ff57187268cc0b4238")
TASKS = (
    "Move_In",
    "Move_Out",
    "Move_Up",
    "Move_Down",
    "Move_Right",
    "Move_Left",
    "Zoom_In",
    "Zoom_Out",
    "Pan_Right",
    "Pan_Left",
    "Tilt_Up",
    "Tilt_Down",
    "Roll_Clockwise",
    "Roll_Counterclockwise",
    "Static",
)
WORKERS = 4
ATTEMPTS = 5
AGENT = {"User-Agent": "video-ai-thesis"}


def _hf(repo: str, revision: str, name: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}"


def _code(name: str) -> str:
    return f"https://raw.githubusercontent.com/{CODE[0]}/{CODE[1]}/camerabench/{name}"


def _fetch(job: tuple[str, Path]) -> str:
    """The file's name once on disk, empty if it already was; written to .part and renamed when whole."""
    url, path = job
    if path.exists():
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    for attempt in range(ATTEMPTS):
        try:
            request = urllib.request.Request(url, headers=AGENT)
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                partial.open("wb") as out,
            ):
                shutil.copyfileobj(response, out, 1 << 20)
            partial.replace(path)
            return path.name
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt + 1 == ATTEMPTS:
                raise RuntimeError(f"{url}: {error}; rerun to resume") from error
            time.sleep(10 * 2**attempt)  # Hugging Face slows anonymous bursts with 429s
    return ""


def _videos() -> list[str]:
    url = f"https://huggingface.co/api/datasets/{VIDEOS[0]}/revision/{VIDEOS[1]}"
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=AGENT), timeout=60
    ) as response:
        meta = json.load(response)
    return sorted(
        s["rfilename"] for s in meta["siblings"] if s["rfilename"].endswith(".mp4")
    )


def main() -> None:
    names = _videos()
    jobs = [
        (_hf(*LABELS, "test.jsonl"), OUT / "test.jsonl"),
        (
            _code("binary_classification_evaluation.py"),
            OUT / "binary_classification_evaluation.py",
        ),
        *(
            (
                _code(f"data/binary_classification/{t}.jsonl"),
                OUT / "tasks" / f"{t}.jsonl",
            )
            for t in TASKS
        ),
        *((_hf(*VIDEOS, name), OUT / "videos" / name) for name in names),
    ]
    with ThreadPoolExecutor(WORKERS) as pool:
        for done, name in enumerate(pool.map(_fetch, jobs), start=1):
            if name:
                print(f"[{done}/{len(jobs)}] {name}", flush=True)
    (OUT / "source.json").write_text(
        json.dumps(
            {
                "videos": {
                    "repo": VIDEOS[0],
                    "revision": VIDEOS[1],
                    "count": len(names),
                },
                "labels": {"repo": LABELS[0], "revision": LABELS[1]},
                "tasks_and_scorer": {"repo": CODE[0], "commit": CODE[1]},
                "license": "labels CC-BY-4.0; clips for research use, not redistributed",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{len(names)} clips and {len(TASKS)} task files in {OUT}")


if __name__ == "__main__":
    main()
