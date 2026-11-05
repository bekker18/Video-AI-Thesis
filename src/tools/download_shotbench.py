"""Downloads ShotBench's shot-size questions and only their images into data/raw/shotbench/:
python3 -m src.tools.download_shotbench"""

from __future__ import annotations

import csv
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "shotbench"
REPO = ("Vchitect/ShotBench", "03c0ce17a9a8c77ecfdd7f4c66c534c23bd7b596")  # pinned
CATEGORY = "shot size"
AGENT = {"User-Agent": "video-ai-thesis"}


def _url(name: str) -> str:
    return f"https://huggingface.co/datasets/{REPO[0]}/resolve/{REPO[1]}/{name}"


def _questions() -> list[dict[str, str]]:
    path = OUT / "test.tsv"
    if not path.exists():
        partial = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(_url("test.tsv"), partial)
        partial.replace(path)
    csv.field_size_limit(10**8)
    with path.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f, delimiter="\t") if r["category"] == CATEGORY]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    questions = _questions()
    wanted = {p for q in questions for p in json.loads(q["path"].replace("'", '"'))}
    missing = {p for p in wanted if not (OUT / p).exists()}
    print(f"{len(questions)} {CATEGORY} questions, {len(missing)} of {len(wanted)} images to fetch", flush=True)
    if missing:
        # The images come in one uncompressed tar: stream it once and keep only the wanted members.
        request = urllib.request.Request(_url("images.tar"), headers=AGENT)
        with urllib.request.urlopen(request, timeout=300) as response, tarfile.open(fileobj=response, mode="r|") as tar:
            for member in tar:
                name = member.name.lstrip("./")
                if name not in missing or not member.isfile():
                    continue
                source = tar.extractfile(member)
                if source is None:
                    continue
                target = OUT / name  # a name from test.tsv, so it cannot leave OUT
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read())
                missing.discard(name)
    (OUT / "source.json").write_text(
        json.dumps(
            {
                "repo": REPO[0],
                "revision": REPO[1],
                "license": "apache-2.0",
                "category": CATEGORY,
                "questions": len(questions),
                "images": len(wanted) - len(missing),
                "not_found_in_tar": sorted(missing),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{len(wanted) - len(missing)} images in {OUT}, {len(missing)} not found", flush=True)


if __name__ == "__main__":
    main()
