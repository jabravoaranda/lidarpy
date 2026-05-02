from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from lidarpy import __version__
from generate_docs_figures import main as generate_docs_figures


ROOT = Path(__file__).resolve().parents[1]
DOCS_SRC = ROOT / "docs"
SITE_DIR = ROOT / "site"


def _copy_static_docs() -> None:
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    SITE_DIR.mkdir(parents=True, exist_ok=True)

    # Copy assets
    assets_src = DOCS_SRC / "assets"
    assets_dst = SITE_DIR / "assets"
    if assets_src.exists():
        shutil.copytree(assets_src, assets_dst)

    # Copy HTML files
    for html_file in DOCS_SRC.glob("*.html"):
        rendered = html_file.read_text(encoding="utf-8").replace(
            "{{VERSION}}", __version__
        )
        (SITE_DIR / html_file.name).write_text(rendered, encoding="utf-8")

    # Disable Jekyll
    (SITE_DIR / ".nojekyll").touch()


def _build_api_reference() -> None:
    api_dir = SITE_DIR / "api"
    api_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [sys.executable, "-m", "pdoc", "-o", str(api_dir), "lidarpy"],
        check=True,
        cwd=str(ROOT),
    )


def main() -> None:
    generate_docs_figures()
    _copy_static_docs()
    _build_api_reference()


if __name__ == "__main__":
    main()
