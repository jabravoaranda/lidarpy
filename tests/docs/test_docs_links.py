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
            if target.startswith("api/"):
                continue

            assert (page.parent / target).exists(), f"{page} links to missing {link}"


def test_static_docs_images_exist():
    for page in DOC_PAGES:
        html = page.read_text(encoding="utf-8")
        images = re.findall(r'src="([^"]+)"', html)

        for image in images:
            target = image.split("#", 1)[0]
            assert (page.parent / target).exists(), f"{page} embeds missing {image}"


def test_static_docs_can_link_to_generated_api_reference():
    index = Path("docs/index.html").read_text(encoding="utf-8")
    workflow = Path(".github/workflows/docs.yml").read_text(encoding="utf-8")
    build_docs = Path("scripts/build_docs.py").read_text(encoding="utf-8")

    assert 'href="api/lidarpy.html"' in index
    assert "python scripts/build_docs.py" in workflow
    assert "generate_docs_figures()" in build_docs
    assert "test -f site/api/lidarpy.html" in workflow
    assert "actions/upload-pages-artifact" in workflow
    assert "actions/deploy-pages" in workflow


def test_reference_docs_mention_core_public_modules():
    references = Path("docs/references.html").read_text(encoding="utf-8")

    assert "lidarpy.nc_convert.measurement" in references
    assert "lidarpy.preprocessing.lidar_preprocessing" in references
    assert "lidarpy.retrieval.synthetic.generator" in references
    assert "lidarpy.retrieval.klett" in references
    assert "lidarpy.retrieval.raman" in references
