"""Utilities for parsing 3GPP release numbers from spec filenames and computing output paths.

3GPP filename convention: ``<spec_number>-<major><minor><editorial>.<ext>``
The major version character encodes the release in base-36:
letters a–z encode releases 10–35 (a=Rel-10, b=Rel-11, …, i=Rel-18, j=Rel-19);
digits 0–9 encode legacy pre-Rel-10 releases.
"""

import re
from pathlib import Path

_REL_SUFFIX_RE = re.compile(r"_rel\d+$")


def parse_3gpp_release(stem: str) -> int | None:
    """Return the 3GPP release number encoded in a spec filename stem, or None.

    Args:
        stem: Filename stem without extension, e.g. ``"38108-i40"``.

    Returns:
        Integer release number (e.g. 18), or ``None`` if not parseable.
    """
    if "-" not in stem:
        return None
    version_part = stem.rsplit("-", 1)[-1]
    if not version_part:
        return None
    major_char = version_part[0].lower()
    if major_char.isalpha():
        return ord(major_char) - ord("a") + 10
    if major_char.isdigit():
        return int(major_char)
    return None


def release_paths(source: Path, data_dir: Path) -> "tuple[Path, Path] | None":
    """Return the canonical (docx_dest, md_dest) paths for a 3GPP spec file.

    Computes:
      - ``<data_dir>/3gpp_rel_<XX>/docx/<stem>_rel<XX>.docx``
      - ``<data_dir>/3gpp_rel_<XX>/md/<stem>_rel<XX>.md``

    Args:
        source: Path to the source ``.docx`` file.
        data_dir: Root data directory (e.g. ``settings.data_dir``).

    Returns:
        A ``(docx_dest, md_dest)`` tuple, or ``None`` if the release cannot
        be parsed from the filename.
    """
    release = parse_3gpp_release(source.stem)
    if release is None:
        return None
    rel_str = f"{release:02d}"
    folder = data_dir / f"3gpp_rel_{rel_str}"
    base_stem = _REL_SUFFIX_RE.sub("", source.stem)
    suffixed_stem = f"{base_stem}_rel{rel_str}"
    return (
        folder / "docx" / f"{suffixed_stem}.docx",
        folder / "md" / f"{suffixed_stem}.md",
    )
