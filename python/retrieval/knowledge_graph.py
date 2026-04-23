# python/retrieval/knowledge_graph.py
"""
Knowledge Graph construction and management using NetworkX.

Builds a directed graph from (Subject, Predicate, Object) triples,
applies Leiden community detection, and provides graph querying utilities.
Persists as GraphML for cross-session use.
"""

import json
import logging
import os

import networkx as nx

logger = logging.getLogger(__name__)

COMMUNITY_SUMMARY_PROMPT = """Summarize the following group of related entities and their relationships in 2-3 sentences.
Focus on what this group is about and how the entities relate to each other.

Entities: {entities}
Relationships: {relationships}

Summary:"""


def _try_leiden_communities(graph: nx.DiGraph) -> dict:
    """
    Attempt Leiden community detection. Falls back to connected components
    if leidenalg/igraph is not available.

    Returns:
        Dict mapping node -> community_id
    """
    try:
        import igraph as ig
        import leidenalg

        # Convert NetworkX DiGraph to undirected igraph for community detection
        undirected = graph.to_undirected()
        # Build igraph from edge list
        nodes = list(undirected.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in undirected.edges()
                 if u in node_to_idx and v in node_to_idx]

        ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)

        # Run Leiden algorithm
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.ModularityVertexPartition
        )

        communities = {}
        for comm_id, members in enumerate(partition):
            for member_idx in members:
                communities[nodes[member_idx]] = comm_id

        logger.info(f"[KG] Leiden detected {len(partition)} communities")
        return communities

    except ImportError:
        logger.warning("[KG] leidenalg/igraph not available, using connected components")
        undirected = graph.to_undirected()
        communities = {}
        for comm_id, component in enumerate(nx.connected_components(undirected)):
            for node in component:
                communities[node] = comm_id
        logger.info(f"[KG] Fallback detected {comm_id + 1} communities (connected components)")
        return communities

    except Exception as e:
        logger.error(f"[KG] Community detection failed: {e}")
        return {node: 0 for node in graph.nodes()}


class KnowledgeGraph:
    """
    A directed knowledge graph built from entity triples.

    Nodes represent entities, edges represent predicates.
    Each node stores its community_id (from Leiden) and source metadata.
    Each edge stores the predicate and source chunk references.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._communities = {}
        self._community_summaries: dict[int, dict] = {}
        self._is_built = False

    @property
    def community_summaries(self) -> dict[int, dict]:
        return getattr(self, "_community_summaries", {}) or {}

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    @property
    def num_communities(self) -> int:
        if not self._communities:
            return 0
        return len(set(self._communities.values()))

    def build_from_triples(self, triples: list[tuple], triple_sources: list[dict] = None) -> None:
        """
        Build the knowledge graph from extracted triples.

        Args:
            triples: List of (subject, predicate, object) tuples
            triple_sources: Optional parallel list of source metadata dicts
        """
        if not triples:
            logger.warning("[KG] No triples provided, graph will be empty")
            self._is_built = True
            return

        if triple_sources is None:
            triple_sources = [{}] * len(triples)

        for i, (subj, pred, obj) in enumerate(triples):
            source = triple_sources[i] if i < len(triple_sources) else {}

            # Add nodes with metadata
            if not self.graph.has_node(subj):
                self.graph.add_node(subj, entity_type="entity", sources=[])
            self.graph.nodes[subj]["sources"].append(source)

            if not self.graph.has_node(obj):
                self.graph.add_node(obj, entity_type="entity", sources=[])
            self.graph.nodes[obj]["sources"].append(source)

            # Add edge with predicate and source
            if self.graph.has_edge(subj, obj):
                # Append predicate if edge already exists
                existing = self.graph.edges[subj, obj]
                predicates = existing.get("predicates", [])
                if pred not in predicates:
                    predicates.append(pred)
                existing["predicates"] = predicates
                existing_sources = existing.get("sources", [])
                existing_sources.append(source)
                existing["sources"] = existing_sources
            else:
                self.graph.add_edge(subj, obj, predicates=[pred], sources=[source])

        self._is_built = True
        logger.info(f"[KG] Built graph: {self.num_nodes} nodes, {self.num_edges} edges")

    def detect_communities(self) -> dict:
        """
        Run Leiden community detection and assign community_id to each node.

        Returns:
            Dict mapping node -> community_id
        """
        if self.num_nodes == 0:
            logger.warning("[KG] Cannot detect communities on empty graph")
            return {}

        self._communities = _try_leiden_communities(self.graph)

        # Assign community to node attributes
        for node, comm_id in self._communities.items():
            if self.graph.has_node(node):
                self.graph.nodes[node]["community_id"] = comm_id

        logger.info(f"[KG] Assigned {self.num_communities} communities to {self.num_nodes} nodes")
        return self._communities

    def generate_community_summaries(self, llm, embedder=None) -> dict[int, dict]:
        """
        LLM summary per Leiden community; optional embedder stores vectors for query-time similarity.
        """
        if not self._communities:
            return {}

        community_ids = sorted(set(self._communities.values()))
        summaries: dict[int, dict] = {}

        for cid in community_ids:
            members = self.get_community_members(cid)
            if len(members) < 2:
                continue

            relationships: list[str] = []
            for u, v, attrs in self.graph.edges(data=True):
                if u in members and v in members:
                    preds = attrs.get("predicates", [])
                    for p in preds[:3]:
                        relationships.append(f"{u} {p} {v}")

            if not relationships:
                continue

            prompt = COMMUNITY_SUMMARY_PROMPT.format(
                entities=", ".join(members[:20]),
                relationships="; ".join(relationships[:15]),
            )

            try:
                summary_text = llm.generate_response(
                    prompt=prompt, temperature=0.3, max_tokens=200
                )
                summary_text = (summary_text or "").strip()
            except Exception as e:
                logger.warning(f"[KG] Community summary LLM failed for community {cid}: {e}")
                summary_text = ""

            if not summary_text:
                continue

            entry: dict = {
                "summary": summary_text,
                "entities": list(members),
                "embedding": None,
            }

            if embedder is not None:
                try:
                    vecs = embedder.encode_text([summary_text], task_type="search_document")
                    if vecs and vecs[0]:
                        entry["embedding"] = list(vecs[0])
                except Exception as e:
                    logger.warning(f"[KG] Community summary embed failed for community {cid}: {e}")

            summaries[cid] = entry

        self._community_summaries = summaries
        logger.info(f"[KG] Generated {len(summaries)} community summaries")
        return summaries

    def get_community_members(self, community_id: int) -> list[str]:
        """Get all entity names in a specific community."""
        return [node for node, cid in self._communities.items() if cid == community_id]

    def get_entity_neighbors(self, entity: str, depth: int = 2) -> list[str]:
        """
        Get all entities within `depth` hops of the given entity.
        Uses BFS on the undirected view.
        """
        if not self.graph.has_node(entity):
            return []

        undirected = self.graph.to_undirected()
        visited = set()
        queue = [(entity, 0)]
        visited.add(entity)

        while queue:
            current, d = queue.pop(0)
            if d >= depth:
                continue
            for neighbor in undirected.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, d + 1))

        visited.discard(entity)  # Don't include the query entity itself
        return list(visited)

    def get_chunks_for_entities(self, entities: list[str]) -> list[dict]:
        """
        Get the source chunk metadata for a set of entities.
        Returns deduplicated list of chunk source dicts.
        """
        chunk_sources = []
        seen_chunks = set()

        for entity in entities:
            if not self.graph.has_node(entity):
                continue
            sources = self.graph.nodes[entity].get("sources", [])
            for source in sources:
                # Create a hashable key for dedup
                key = (source.get("source", ""), source.get("page", ""),
                       source.get("chunk_index", ""))
                if key not in seen_chunks:
                    seen_chunks.add(key)
                    chunk_sources.append(source)

        return chunk_sources

    def find_entity(self, query_term: str) -> list[str]:
        """
        Find entities in the graph that match or contain the query term.
        Case-insensitive partial matching.
        """
        query_lower = query_term.lower().strip()
        matches = []
        for node in self.graph.nodes():
            if query_lower in node.lower() or node.lower() in query_lower:
                matches.append(node)
        return matches

    def save(self, path: str) -> None:
        """
        Persist the knowledge graph to disk.
        Saves as JSON (GraphML doesn't handle list attributes well).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Serialize to a JSON-friendly format
        data = {
            "nodes": {},
            "edges": [],
            "communities": self._communities,
            "community_summaries": {
                str(k): v for k, v in getattr(self, "_community_summaries", {}).items()
            },
        }

        for node, attrs in self.graph.nodes(data=True):
            # Convert sources to serializable format
            serializable_attrs = {}
            for k, v in attrs.items():
                serializable_attrs[k] = v
            data["nodes"][node] = serializable_attrs

        for u, v, attrs in self.graph.edges(data=True):
            edge_data = {"source": u, "target": v}
            for k, val in attrs.items():
                edge_data[k] = val
            data["edges"].append(edge_data)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"[KG] Graph saved to {path} ({self.num_nodes} nodes, {self.num_edges} edges)")

    def load(self, path: str) -> bool:
        """Load a knowledge graph from a JSON file. Returns True if successful."""
        if not os.path.exists(path):
            logger.warning(f"[KG] Graph file not found: {path}")
            return False

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.graph = nx.DiGraph()

            # Restore nodes
            for node, attrs in data.get("nodes", {}).items():
                self.graph.add_node(node, **attrs)

            # Restore edges
            for edge in data.get("edges", []):
                u = edge.pop("source")
                v = edge.pop("target")
                self.graph.add_edge(u, v, **edge)

            # Restore communities
            self._communities = data.get("communities", {})

            raw_summaries = data.get("community_summaries", {}) or {}
            self._community_summaries = {}
            for k, v in raw_summaries.items():
                try:
                    cid = int(k)
                except (ValueError, TypeError):
                    continue
                if isinstance(v, dict):
                    self._community_summaries[cid] = v

            self._is_built = True
            logger.info(f"[KG] Graph loaded from {path}: {self.num_nodes} nodes, "
                         f"{self.num_edges} edges, {self.num_communities} communities")
            return True

        except Exception as e:
            logger.error(f"[KG] Failed to load graph from {path}: {e}")
            return False

    def get_summary(self) -> dict:
        """Return a summary of the graph for logging/debugging."""
        return {
            "nodes": self.num_nodes,
            "edges": self.num_edges,
            "communities": self.num_communities,
            "top_entities": sorted(
                self.graph.nodes(),
                key=lambda n: self.graph.degree(n),
                reverse=True
            )[:10],
        }
