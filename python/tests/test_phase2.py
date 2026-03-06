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

    def generate_response(self, prompt="", max_tokens=500, temperature=0.1, messages=None):
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

        triples, sources = extractor.extract_from_chunks(chunks)
        assert len(triples) > 0
        assert len(triples) == len(sources)

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
            {"page": 1, "source": "test.pdf", "chunk_index": 0, "chunk_text_preview": "ML text"}
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

        save_path = str(tmp_path / "test_graph.json")
        kg.save(save_path)

        loaded = KnowledgeGraph()
        assert loaded.load(save_path)
        assert loaded.num_nodes == kg.num_nodes
        assert loaded.num_edges == kg.num_edges

    def test_empty_graph(self):
        from retrieval.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.build_from_triples([])

        assert kg.num_nodes == 0
        assert kg.num_edges == 0
        assert kg.get_entity_neighbors("anything") == []


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
            {"page": 1, "source": "test.pdf", "chunk_index": 0, "chunk_text_preview": "ML is a subset of AI"},
            {"page": 1, "source": "test.pdf", "chunk_index": 1, "chunk_text_preview": "DL is a subset of ML"},
            {"page": 2, "source": "test.pdf", "chunk_index": 2, "chunk_text_preview": "NNs are used for DL"},
        ]
        kg.build_from_triples(triples, sources)
        return kg

    def test_score_chunks(self, kg_with_data):
        from retrieval.graph_retrieval import GraphRetriever

        retriever = GraphRetriever(kg_with_data, llm=None, traversal_depth=2)
        results = retriever.score_chunks_by_graph("What is machine learning?")

        assert len(results) > 0
        assert all(r["retrieval_method"] == "graph" for r in results)

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
