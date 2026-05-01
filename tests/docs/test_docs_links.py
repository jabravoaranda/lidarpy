from __future__ import annotations

import re
from pathlib import Path


DOC_PAGES = [
    Path("docs/index.html"),
    Path("docs/getting-started.html"),
    Path("docs/processing-workflow.html"),
    Path("docs/examples.html"),
    Path("docs/references.html"),
    Path("docs/contributing.html"),
]


def test_readme_links_to_existing_docs():
    readme = Path("README.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\((docs/[^)]+)\)", readme)

    assert links
    for link in links:
        assert Path(link).exists()


def test_static_docs_pages_exist():
    for page in DOC_PAGES:
        html = page.read_text(encoding="utf-8")

        assert "<!doctype html>" in html.lower()
        assert "<title>" in html
        assert 'href="assets/site.css"' in html


def test_static_docs_links_are_local_and_existing():
    assert Path("docs/assets/site.css").exists()

    for page in DOC_PAGES:
        html = page.read_text(encoding="utf-8")
        links = re.findall(r'href="([^"]+)"', html)

        assert links
        for link in links:
            if link.startswith(("http://", "https://", "#")):
                continue

            target = link.split("#", 1)[0]
            assert (page.parent / target).exists(), f"{page} links to missing {link}"


def test_reference_docs_mention_core_public_modules():
    references = Path("docs/references.html").read_text(encoding="utf-8")

    assert "lidarpy.nc_convert.measurement" in references
    assert "lidarpy.preprocessing.lidar_preprocessing" in references
    assert "lidarpy.retrieval.synthetic.generator" in references
    assert "lidarpy.retrieval.klett" in references
    assert "lidarpy.retrieval.raman" in references
