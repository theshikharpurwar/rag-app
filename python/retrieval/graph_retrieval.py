# python/retrieval/graph_retrieval.py
"""
Graph-based retrieval scoring.

Given a query, identifies relevant entities in the Knowledge Graph,
traverses their neighborhoods via BFS, and scores chunks by proximity
to query-relevant entities.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Prompt for extracting query entities
ENTITY_EXTRACTION_PROMPT = """Extract the key entities (nouns, concepts, names, methods, technologies) from this query.
Return ONLY a JSON array of strings. No explanations.

Query: {query}

RESPONSE (JSON array only):"""


class GraphRetriever:
    """
    Retrieves and scores chunks using Knowledge Graph traversal.

    Workflow:
    1. Extract entities from query (via LLM or keyword matching)
    2. Find matching entities in the Knowledge Graph
    3. BFS traverse from matched entities
    4. Score chunks by graph distance to query entities
    """

    def __init__(self, knowledge_graph, llm=None, traversal_depth: int = 2):
        """
        Args:
            knowledge_graph: A KnowledgeGraph instance
            llm: Optional OllamaLLM for entity extraction from queries
            traversal_depth: BFS depth limit for neighbor discovery
        """
        self.kg = knowledge_graph
        self.llm = llm
        self.traversal_depth = traversal_depth

    def extract_query_entities(self, query: str) -> list[str]:
        """
        Extract key entities from a query.
        Uses keyword matching against graph nodes first (instant),
        falls back to LLM only when no keyword matches are found.
        """
        entities = []

        # Method 1: Keyword matching — instant, often sufficient
        query_words = set(w.strip('.,;:!?\'"()[]{}') for w in query.lower().split())
        query_words.discard('')  # remove any empty strings from stripping
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how',
                      'why', 'when', 'where', 'who', 'which', 'this', 'that',
                      'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
                      'with', 'by', 'from', 'does', 'do', 'did', 'can', 'could',
                      'should', 'would', 'will', 'about', 'it', 'its', 'they',
                      'tell', 'me', 'his', 'her', 'my', 'your', 'our', 'their'}
        query_words -= stop_words

        for node in self.kg.graph.nodes():
            node_words = set(node.lower().split())
            # If any query word matches a word in the node name
            if query_words & node_words:
                entities.append(node)

        if entities:
            logger.info(f"[GraphRetriever] Keyword match: {len(entities)} entities: {entities[:5]}")
            return entities

        # Method 2: LLM-based extraction (fallback — only when keyword fails)
        if self.llm:
            try:
                prompt = ENTITY_EXTRACTION_PROMPT.format(query=query)
                response = self.llm.generate_response(
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.1
                )

                if response:
                    import json
                    # Try to parse JSON array
                    try:
                        parsed = json.loads(response.strip())
                        if isinstance(parsed, list):
                            entities = [str(e).lower().strip() for e in parsed if e]
                    except json.JSONDecodeError:
                        # Try to find array in response
                        match = re.search(r'\[.*?\]', response, re.DOTALL)
                        if match:
                            try:
                                parsed = json.loads(match.group())
                                entities = [str(e).lower().strip() for e in parsed if e]
                            except json.JSONDecodeError:
                                pass

                if entities:
                    logger.info(f"[GraphRetriever] LLM fallback: {len(entities)} entities: {entities[:5]}")
            except Exception as e:
                logger.warning(f"[GraphRetriever] LLM entity extraction failed: {e}")

        if not entities:
            logger.info(f"[GraphRetriever] No entities extracted for query: {query[:50]}")

        return entities

    def score_chunks_by_graph(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Score and retrieve chunks based on graph proximity to query entities.

        Args:
            query: The user's question
            top_k: Maximum number of chunks to return

        Returns:
            List of scored chunk dicts with keys: text, page, source, score, retrieval_method
        """
        if self.kg.num_nodes == 0:
            logger.warning("[GraphRetriever] Knowledge graph is empty, no graph results")
            return []

        # 1. Extract entities from query
        query_entities = self.extract_query_entities(query)
        if not query_entities:
            logger.info("[GraphRetriever] No query entities extracted, falling back to keyword match")
            return []

        # 2. Find matching entities in the graph
        matched_entities = []
        for qe in query_entities:
            matches = self.kg.find_entity(qe)
            matched_entities.extend(matches)

        # Deduplicate
        matched_entities = list(set(matched_entities))

        if not matched_entities:
            logger.info(f"[GraphRetriever] No matching entities found in graph for: {query_entities}")
            return []

        logger.info(f"[GraphRetriever] Matched {len(matched_entities)} entities in graph: "
                     f"{matched_entities[:5]}")

        # 3. BFS from matched entities to find related entities
        related_entities = set(matched_entities)
        entity_distances = {e: 0 for e in matched_entities}  # distance from query entities

        for entity in matched_entities:
            neighbors = self.kg.get_entity_neighbors(entity, depth=self.traversal_depth)
            for neighbor in neighbors:
                if neighbor not in related_entities:
                    # Calculate approximate distance
                    related_entities.add(neighbor)
                    if neighbor not in entity_distances:
                        entity_distances[neighbor] = self.traversal_depth  # approximate

        # Calculate more precise distances via BFS
        undirected = self.kg.graph.to_undirected()
        for entity in matched_entities:
            try:
                distances = dict(
                    level for level in _bfs_levels(undirected, entity, self.traversal_depth)
                )
                for node, dist in distances.items():
                    if node not in entity_distances or dist < entity_distances[node]:
                        entity_distances[node] = dist
            except Exception:
                pass

        # 4. Score chunks by proximity to related entities
        chunk_scores = {}  # chunk_key -> {"score": float, "chunk_data": dict}

        for entity in related_entities:
            distance = entity_distances.get(entity, self.traversal_depth)
            # Score: closer entities get higher scores
            entity_score = 1.0 / (1.0 + distance)

            chunk_sources = self.kg.get_chunks_for_entities([entity])
            for source in chunk_sources:
                chunk_key = (
                    source.get("source", ""),
                    source.get("page", ""),
                    source.get("chunk_index", "")
                )
                if chunk_key in chunk_scores:
                    chunk_scores[chunk_key]["score"] += entity_score
                else:
                    chunk_scores[chunk_key] = {
                        "score": entity_score,
                        "text": source.get("chunk_text") or source.get("chunk_text_preview", ""),
                        "page": source.get("page", "N/A"),
                        "source": source.get("source", "Unknown"),
                        "chunk_index": source.get("chunk_index", 0),
                    }

        # 5. Convert to list and sort
        results = []
        for chunk_key, chunk_data in chunk_scores.items():
            results.append({
                "text": chunk_data["text"],
                "page": chunk_data["page"],
                "source": chunk_data["source"],
                "chunk_index": chunk_data["chunk_index"],
                "score": chunk_data["score"],
                "retrieval_method": "graph",
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"[GraphRetriever] Scored {len(results)} chunks, returning top {top_k}")
        return results[:top_k]


def _bfs_levels(graph, start, max_depth):
    """
    BFS yielding (node, depth) tuples up to max_depth.
    """
    visited = {start}
    queue = [(start, 0)]

    while queue:
        node, depth = queue.pop(0)
        yield node, depth

        if depth >= max_depth:
            continue

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
