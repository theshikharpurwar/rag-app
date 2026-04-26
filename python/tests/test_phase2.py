# python/tests/test_phase2.py
"""
Unit tests for Phase 2: Hybrid Retrieval & Knowledge Graphs.

Tests BM25 search, entity extraction (with mocked LLM), knowledge graph
construction/communities, graph retrieval scoring, and RRF fusion.

Run: cd python && python -m pytest tests/test_phase2.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the python directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# =============================================================================
# BM25 Search Tests
# =============================================================================

class TestBM25Index:
    """Tests for BM25 keyword search."""

    @pytest.fixture
    def sample_chunks(self):
        return [
            {"text": "Machine learning is a subset of artificial intelligence.",
             "pdf_id": "doc1", "page": 1, "source": "test.pdf", "chunk_index": 0},
            {"text": "Neural networks are used for deep learning tasks.",
             "pdf_id": "doc1", "page": 1, "source": "test.pdf", "chunk_index": 1},
            {"text": "Natural language processing handles text data analysis.",
             "pdf_id": "doc1", "page": 2, "source": "test.pdf", "chunk_index": 2},
            {"text": "The knowledge graph organizes entities and relationships.",
             "pdf_id": "doc2", "page": 1, "source": "other.pdf", "chunk_index": 0},
        ]

    def test_build_and_search(self, sample_chunks):
        from retrieval.bm25_search import BM25Index

        idx = BM25Index()
        idx.build_index(sample_chunks)
        results = idx.search("machine learning artificial intelligence")

        assert len(results) > 0
        assert results[0]["retrieval_method"] == "bm25"
        # The first result should be the one about machine learning
        assert "machine learning" in results[0]["text"].lower()

    def test_pdf_id_filter(self, sample_chunks):
        from retrieval.bm25_search import BM25Index

        idx = BM25Index()
        idx.build_index(sample_chunks)
        results = idx.search("knowledge graph", pdf_id="doc2")

        assert len(results) > 0
        assert all(r["source"] == "other.pdf" for r in results)

    def test_save_and_load(self, sample_chunks, tmp_path):
        from retrieval.bm25_search import BM25Index

        idx = BM25Index()
        idx.build_index(sample_chunks)
        save_path = str(tmp_path / "test_bm25.pkl")
        idx.save(save_path)

        loaded = BM25Index()
        assert loaded.load(save_path)
        results = loaded.search("neural networks")
        assert len(results) > 0

    def test_empty_index(self):
        from retrieval.bm25_search import BM25Index

        idx = BM25Index()
        idx.build_index([])
        results = idx.search("anything")
        assert results == []

    def test_empty_query(self, sample_chunks):
        from retrieval.bm25_search import BM25Index

        idx = BM25Index()
        idx.build_index(sample_chunks)
        results = idx.search("")
        assert results == []


# =============================================================================
# Entity Extraction Tests
# =============================================================================

class MockLLM:
    """Mock LLM that returns predictable triple outputs."""

    def generate_response(self, prompt="", max_tokens=500, temperature=0.1, messages=None, **kwargs):
        if "Extract" in prompt or "extract" in prompt:
            return json.dumps([
                ["machine learning", "is_subset_of", "artificial intelligence"],
                ["neural networks", "used_for", "deep learning"],
                ["RAG system", "uses", "knowledge graphs"],
            ])
        if "entities" in prompt.lower() or "Extract" in prompt:
            return json.dumps(["machine learning", "neural networks", "artificial intelligence"])
        return "[]"


class TestEntityExtractor:
    """Tests for LLM-powered entity extraction."""

    def test_extract_triples(self):
        from retrieval.entity_extractor import EntityExtractor

        extractor = EntityExtractor(MockLLM())
        triples = extractor.extract_triples("Machine learning is a subset of AI.")

        assert len(triples) > 0
        # Each triple should be (subject, predicate, object)
        for s, p, o in triples:
            assert isinstance(s, str)
            assert isinstance(p, str)
            assert isinstance(o, str)
            assert len(s) > 0
            assert len(p) > 0
            assert len(o) > 0

    def test_extract_from_chunks(self):
        from retrieval.entity_extractor import EntityExtractor

        extractor = EntityExtractor(MockLLM())
        chunks = [
            {"text": "ML is great.", "page": 1, "source": "test.pdf", "chunk_index": 0},
            {"text": "AI is powerful.", "page": 2, "source": "test.pdf", "chunk_index": 1},
        ]

        triples, sources, audit = extractor.extract_from_chunks(chunks)
        assert len(triples) > 0
        assert len(triples) == len(sources)
        assert audit["mode"] == "llm_triples"
        assert audit["chunks_processed"] == 2

    def test_empty_input(self):
        from retrieval.entity_extractor import EntityExtractor

        extractor = EntityExtractor(MockLLM())
        triples = extractor.extract_triples("")
        assert triples == []

    def test_entity_normalization(self):
        from retrieval.entity_extractor import _normalize_entity

        assert _normalize_entity("The Algorithm") == "algorithm"
        assert _normalize_entity("  A  Neural  Network  ") == "neural network"
        assert _normalize_entity("an entity.") == "entity"

    def test_triple_quality_golden(self):
        from retrieval.entity_extractor import EntityExtractor

        extractor = EntityExtractor(MockLLM(), batch_size=1, max_chars=1200)
        triples, _, _ = extractor.extract_from_chunks(
            [{"text": "Machine learning is a subset of artificial intelligence.", "page": 1, "source": "g.pdf", "chunk_index": 0}]
        )

        assert ("machine learning", "is_subset_of", "artificial intelligence") in triples


class TestCooccurrenceExtractor:
    def test_deterministic_edges(self):
        from retrieval.entity_extractor import NounPhraseCooccurrenceExtractor

        chunks = [
            {
                "text": "Graph Neural Network uses BM25 and queryRouter with LLM and RAG.",
                "page": 1,
                "source": "test.pdf",
                "chunk_index": 0,
            }
        ]
        extractor = NounPhraseCooccurrenceExtractor("regex")
        triples1, _, audit1 = extractor.extract_from_chunks(chunks)
        triples2, _, audit2 = extractor.extract_from_chunks(chunks)

        assert triples1 == triples2
        assert audit1["triples_after_dedup"] == audit2["triples_after_dedup"]
        assert all(t[1] == "co_occurs_with" for t in triples1)

    def test_spacy_stub_raises(self):
        from retrieval.entity_extractor import NounPhraseCooccurrenceExtractor

        with pytest.raises(NotImplementedError):
            NounPhraseCooccurrenceExtractor("spacy")


def test_kg_audit_artifact_shape(tmp_path, monkeypatch):
    import compute_embeddings

    monkeypatch.setattr(compute_embeddings, "INDICES_DIR", str(tmp_path))
    payload = {
        "pdf_id": "abc",
        "kg_mode": "llm_triples",
        "graph_summary": {"nodes": 1, "edges": 2, "communities": 1},
        "graph_json_size_bytes": 123,
        "timings": {"extraction_seconds": 1.2, "graph_build_seconds": 0.1, "community_detection_seconds": 0.2, "save_seconds": 0.05},
        "extraction_audit": {
            "mode": "llm_triples",
            "batches": [{"response": "not-json", "parser_errors": 1}],
            "parser_error_count": 1,
            "truncation_count": 0,
        },
    }
    out = compute_embeddings._write_kg_audit_artifact("abc", payload)
    assert out is not None
    data = json.loads(Path(out).read_text(encoding="utf-8"))
    assert data["pdf_id"] == "abc"
    assert "graph_summary" in data
    assert "timings" in data
    assert data["extraction_audit"]["batches"][0]["response"] == "not-json"


# =============================================================================
# Knowledge Graph Tests
# =============================================================================

class TestKnowledgeGraph:
    """Tests for NetworkX knowledge graph construction and querying."""

    @pytest.fixture
    def sample_triples(self):
        return [
            ("machine learning", "is_subset_of", "artificial intelligence"),
            ("deep learning", "is_subset_of", "machine learning"),
            ("neural network", "used_for", "deep learning"),
            ("transformer", "is_a", "neural network"),
            ("bert", "is_a", "transformer"),
            ("rag system", "uses", "knowledge graph"),
            ("knowledge graph", "stores", "entities"),
        ]

    @pytest.fixture
    def sample_sources(self, sample_triples):
        return [
            {"page": 1, "source": "test.pdf", "chunk_index": 0, "chunk_text": "ML text"}
        ] * len(sample_triples)

    def test_build_graph(self, sample_triples, sample_sources):
        from retrieval.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.build_from_triples(sample_triples, sample_sources)

        assert kg.num_nodes > 0
        assert kg.num_edges > 0
        assert kg.num_nodes == 9  # 9 unique entities
        assert kg.num_edges == 7  # 7 unique edges

    def test_community_detection(self, sample_triples, sample_sources):
        from retrieval.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.build_from_triples(sample_triples, sample_sources)
        communities = kg.detect_communities()

        assert len(communities) > 0
        assert kg.num_communities >= 1

    def test_entity_neighbors(self, sample_triples, sample_sources):
        from retrieval.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.build_from_triples(sample_triples, sample_sources)

        neighbors = kg.get_entity_neighbors("machine learning", depth=1)
        assert "artificial intelligence" in neighbors
        assert "deep learning" in neighbors

    def test_find_entity(self, sample_triples, sample_sources):
        from retrieval.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.build_from_triples(sample_triples, sample_sources)

        matches = kg.find_entity("neural")
        assert "neural network" in matches

    def test_save_and_load(self, sample_triples, sample_sources, tmp_path):
        from retrieval.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.build_from_triples(sample_triples, sample_sources)
        kg.detect_communities()
        kg.set_metadata(kg_mode="llm_triples", storage_pretty=False)

        save_path = str(tmp_path / "test_graph.json")
        kg.save(save_path)

        loaded = KnowledgeGraph()
        assert loaded.load(save_path)
        assert loaded.num_nodes == kg.num_nodes
        assert loaded.num_edges == kg.num_edges
        assert loaded.get_mode() == "llm_triples"
        assert loaded.get_chunks_for_entities(["machine learning"])[0]["chunk_text"] == "ML text"

    def test_empty_graph(self):
        from retrieval.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.build_from_triples([])

        assert kg.num_nodes == 0
        assert kg.num_edges == 0
        assert kg.get_entity_neighbors("anything") == []

    def test_mode_mismatch_returns_false(self, sample_triples, sample_sources, tmp_path):
        from retrieval.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.build_from_triples(sample_triples, sample_sources)
        kg.set_metadata(kg_mode="noun_phrase_cooccurrence", storage_pretty=False)
        save_path = str(tmp_path / "mismatch_graph.json")
        kg.save(save_path)

        loaded = KnowledgeGraph()
        assert loaded.load(save_path, expected_mode="llm_triples") is False

    def test_compact_storage_smaller_than_pretty(self, sample_triples, sample_sources, tmp_path):
        from retrieval.knowledge_graph import KnowledgeGraph

        pretty_path = tmp_path / "pretty_graph.json"
        compact_path = tmp_path / "compact_graph.json"

        pretty = KnowledgeGraph()
        pretty.build_from_triples(sample_triples, sample_sources)
        pretty.set_metadata(kg_mode="llm_triples", storage_pretty=True)
        pretty.save(str(pretty_path))

        compact = KnowledgeGraph()
        compact.build_from_triples(sample_triples, sample_sources)
        compact.set_metadata(kg_mode="llm_triples", storage_pretty=False)
        compact.save(str(compact_path))

        assert compact_path.stat().st_size < pretty_path.stat().st_size


# =============================================================================
# RRF Fusion Tests
# =============================================================================

class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion."""

    def test_basic_fusion(self):
        from retrieval.rrf_fusion import rrf_fuse

        list1 = [
            {"text": "doc A about ML", "page": 1, "source": "a.pdf", "score": 0.9, "retrieval_method": "vector"},
            {"text": "doc B about AI", "page": 2, "source": "a.pdf", "score": 0.7, "retrieval_method": "vector"},
        ]
        list2 = [
            {"text": "doc B about AI", "page": 2, "source": "a.pdf", "score": 5.0, "retrieval_method": "bm25"},
            {"text": "doc C about NLP", "page": 3, "source": "a.pdf", "score": 3.0, "retrieval_method": "bm25"},
        ]

        fused = rrf_fuse([list1, list2], weights=[0.6, 0.4], top_k=5)

        assert len(fused) > 0
        # Doc B should rank higher since it appears in both lists
        doc_b_results = [r for r in fused if "doc B" in r["text"]]
        assert len(doc_b_results) > 0
        assert len(doc_b_results[0]["retrieval_methods"]) == 2  # appeared in both

    def test_equal_weights(self):
        from retrieval.rrf_fusion import rrf_fuse

        list1 = [{"text": "only in vector", "page": 1, "source": "a.pdf",
                   "score": 1.0, "retrieval_method": "vector"}]
        list2 = [{"text": "only in bm25", "page": 2, "source": "a.pdf",
                   "score": 1.0, "retrieval_method": "bm25"}]

        fused = rrf_fuse([list1, list2])
        assert len(fused) == 2
        # With equal weights and same rank, scores should be equal
        assert abs(fused[0]["score"] - fused[1]["score"]) < 0.01

    def test_empty_lists(self):
        from retrieval.rrf_fusion import rrf_fuse

        assert rrf_fuse([]) == []
        assert rrf_fuse([[], []]) == []

    def test_single_list(self):
        from retrieval.rrf_fusion import rrf_fuse

        results = [{"text": "single", "page": 1, "source": "a.pdf",
                     "score": 1.0, "retrieval_method": "vector"}]
        fused = rrf_fuse([results])
        assert len(fused) == 1


# =============================================================================
# Graph Retrieval Tests
# =============================================================================

class TestGraphRetriever:
    """Tests for graph-based retrieval scoring."""

    @pytest.fixture
    def kg_with_data(self):
        from retrieval.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        triples = [
            ("machine learning", "is_subset_of", "artificial intelligence"),
            ("deep learning", "is_subset_of", "machine learning"),
            ("neural network", "used_for", "deep learning"),
        ]
        sources = [
            {
                "page": 1,
                "source": "test.pdf",
                "chunk_index": 0,
                "chunk_text": (
                    "ML is a subset of AI. "
                    "This sentence is intentionally long to ensure graph retrieval keeps full chunk text "
                    "instead of falling back to a truncated preview."
                ),
            },
            {"page": 1, "source": "test.pdf", "chunk_index": 1, "chunk_text": "DL is a subset of ML"},
            {"page": 2, "source": "test.pdf", "chunk_index": 2, "chunk_text": "NNs are used for DL"},
        ]
        kg.build_from_triples(triples, sources)
        return kg

    def test_score_chunks(self, kg_with_data):
        from retrieval.graph_retrieval import GraphRetriever

        retriever = GraphRetriever(kg_with_data, llm=None, traversal_depth=2)
        results = retriever.score_chunks_by_graph("What is machine learning?")

        assert len(results) > 0
        assert all(r["retrieval_method"] == "graph" for r in results)
        assert any(
            "This sentence is intentionally long to ensure graph retrieval keeps full chunk text" in r["text"]
            for r in results
        )

    def test_no_match(self, kg_with_data):
        from retrieval.graph_retrieval import GraphRetriever

        retriever = GraphRetriever(kg_with_data, llm=None, traversal_depth=2)
        results = retriever.score_chunks_by_graph("quantum computing blockchain")

        # Should return empty since those entities don't exist in the graph
        assert len(results) == 0

    def test_with_mock_llm(self, kg_with_data):
        from retrieval.graph_retrieval import GraphRetriever

        retriever = GraphRetriever(kg_with_data, llm=MockLLM(), traversal_depth=2)
        results = retriever.score_chunks_by_graph("Tell me about neural networks")

        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
