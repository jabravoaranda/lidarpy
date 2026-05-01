from __future__ import annotations

import re
from pathlib import Path


def test_readme_links_to_existing_docs():
    readme = Path("README.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\((docs/[^)]+)\)", readme)

    assert links
    for link in links:
        assert Path(link).exists()


def test_reference_docs_mention_core_public_modules():
    references = Path("docs/references.md").read_text(encoding="utf-8")

    assert "lidarpy.nc_convert.measurement" in references
    assert "lidarpy.preprocessing.lidar_preprocessing" in references
    assert "lidarpy.retrieval.synthetic.generator" in references
    assert "lidarpy.retrieval.klett" in references
    assert "lidarpy.retrieval.raman" in references
