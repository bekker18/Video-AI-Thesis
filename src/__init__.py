"""Pipeline stages, numbered on disk and imported by plain name.

Stage directories carry their position in the pipeline (01-sampling,
02-segmentation, 03-association, ...) so the order is visible in the file tree.
Those names are not valid module names, though, so `import src.sampling` cannot
find them on its own. The finder below maps each stage's plain name onto its
numbered directory, which keeps the numbering where it is useful and the
imports readable:

    python -m src.association       -> src/03-association/__main__.py
    from src.sampling.sampler ...   -> src/01-sampling/sampler.py

Stages whose directory has no __init__.py are not written yet and stay
unimportable, exactly as they would without the mapping.
"""

from __future__ import annotations

import sys
from importlib.util import spec_from_file_location
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def _discover_stages() -> dict[str, Path]:
    """Plain stage name -> numbered directory."""
    stages: dict[str, Path] = {}
    for directory in sorted(_ROOT.iterdir()):
        number, _, name = directory.name.partition("-")
        if not (number.isdigit() and name):
            continue
        if not (directory / "__init__.py").exists():
            continue
        # Multi-word stages are hyphenated on disk, underscored when imported.
        stages[name.replace("-", "_")] = directory
    return stages


_STAGES = _discover_stages()


class _StageFinder:
    """Resolves src.<stage> to the numbered directory that holds it."""

    _prefix = f"{__name__}."

    @classmethod
    def find_spec(cls, fullname: str, path=None, target=None):
        if not fullname.startswith(cls._prefix):
            return None
        directory = _STAGES.get(fullname[len(cls._prefix) :])
        if directory is None:
            return None
        # Submodules resolve through the normal machinery from here, since the
        # spec carries the numbered directory as the package's search path.
        return spec_from_file_location(
            fullname,
            directory / "__init__.py",
            submodule_search_locations=[str(directory)],
        )


# Appended rather than prepended, so a stage alias can never shadow a real
# module that the standard finders would have resolved first.
if _StageFinder not in sys.meta_path:
    sys.meta_path.append(_StageFinder)

__all__ = sorted(_STAGES)
