import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_visualize_kg_generates_html_and_png(tmp_path, capsys):
    pytest.importorskip("pyvis")
    pytest.importorskip("matplotlib")

    from evaluation import visualize_kg
    from retrieval.knowledge_graph import KnowledgeGraph

    triples = [
        ("machine learning", "is_subset_of", "artificial intelligence"),
        ("deep learning", "is_subset_of", "machine learning"),
        ("neural network", "used_for", "deep learning"),
    ]
    sources = [
        {"page": 1, "source": "fixture.pdf", "chunk_index": 0, "chunk_text": "ML relation"},
        {"page": 1, "source": "fixture.pdf", "chunk_index": 1, "chunk_text": "DL relation"},
        {"page": 2, "source": "fixture.pdf", "chunk_index": 2, "chunk_text": "NN relation"},
    ]

    kg = KnowledgeGraph()
    kg.build_from_triples(triples, sources)
    kg.detect_communities()
    kg_path = tmp_path / "foo_graph.json"
    kg.save(str(kg_path))

    output_dir = tmp_path / "out"
    rc = visualize_kg.main(
        [
            "--pdf_id",
            "foo",
            "--indices-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--format",
            "both",
            "--max-nodes",
            "200",
        ]
    )
    assert rc == 0

    stdout = capsys.readouterr().out.strip()
    payload = json.loads(stdout)

    html_path = output_dir / "foo_kg.html"
    png_path = output_dir / "foo_kg.png"

    assert payload["html"] == str(html_path)
    assert payload["png"] == str(png_path)
    assert payload["num_nodes"] > 0
    assert payload["num_edges"] > 0

    assert html_path.exists()
    assert png_path.exists()
    assert html_path.stat().st_size > 0
    assert png_path.stat().st_size > 0
