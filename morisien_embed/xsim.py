from __future__ import annotations

import json
import urllib.request
from pathlib import Path

XSIM_BASE = "https://dl.fbaipublicfiles.com/nllb/laser/xsimplusplus"
ERRTYPE_FILE = "eng_Latn_errtype.devtest.json"


def fetch_errtype(cache_dir: Path) -> dict[str, dict[str, str]]:
    """Map each augmented English sentence to the rule that made it and the sentence it came from.

    Args:
        cache_dir (`Path`): Directory the file is downloaded into once and read from after.

    Returns:
        `dict[str, dict[str, str]]`: Augmented sentence to `{"errtype": ..., "src": ...}`.
    """
    path = cache_dir / ERRTYPE_FILE
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{XSIM_BASE}/{ERRTYPE_FILE}", timeout=300) as response:
            path.write_bytes(response.read())
    return json.loads(path.read_text(encoding="utf-8"))
