# python/retrieval/entity_extractor.py
"""
LLM-powered entity and relationship extraction for Knowledge Graph construction.

Sends each text chunk to the LLM with a structured prompt to extract
(Subject, Predicate, Object) triples. Includes basic entity resolution
(normalization, deduplication).
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# Prompt template for triple extraction
EXTRACTION_PROMPT = """You are an expert at extracting structured knowledge from text.
Extract all important entities and their relationships from the following text.

Return ONLY a JSON array of triples, where each triple is [subject, predicate, object].
- Subject and Object should be noun phrases (people, concepts, methods, technologies, etc.)
- Predicate should be the relationship between them (uses, is_a, part_of, causes, etc.)
- Extract at least 1 triple if any relationship exists, up to 10 triples maximum.
- If no meaningful relationships exist, return an empty array: []

TEXT:
{text}

RESPONSE (JSON array only, no markdown, no explanation):"""


def _normalize_entity(entity: str) -> str:
    """
    Normalize an entity string for deduplication.
    - Lowercase
    - Strip leading articles (the, a, an)
    - Strip extra whitespace
    - Remove trailing punctuation
    """
    entity = entity.strip().lower()
    # Remove leading articles
    entity = re.sub(r'^(the|a|an)\s+', '', entity)
    # Remove trailing punctuation
    entity = entity.rstrip('.,;:!?')
    # Collapse whitespace
    entity = re.sub(r'\s+', ' ', entity)
    return entity.strip()


def _parse_triples_response(response: str) -> list[tuple]:
    """
    Parse LLM response into a list of (subject, predicate, object) tuples.
    Handles various response formats robustly.
    """
    triples = []

    # Try to extract JSON array from the response
    # First, try direct JSON parse
    try:
        parsed = json.loads(response.strip())
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    triples.append((str(item[0]), str(item[1]), str(item[2])))
                elif isinstance(item, dict):
                    s = item.get('subject', item.get('s', ''))
                    p = item.get('predicate', item.get('p', item.get('relation', '')))
                    o = item.get('object', item.get('o', ''))
                    if s and p and o:
                        triples.append((str(s), str(p), str(o)))
            return triples
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the response using regex (greedy to get outer array)
    json_match = re.search(r'\[[\s\S]*\]', response)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        triples.append((str(item[0]), str(item[1]), str(item[2])))
                    elif isinstance(item, dict):
                        s = item.get('subject', item.get('s', ''))
                        p = item.get('predicate', item.get('p', item.get('relation', '')))
                        o = item.get('object', item.get('o', ''))
                        if s and p and o:
                            triples.append((str(s), str(p), str(o)))
        except json.JSONDecodeError:
            pass

    return triples


class EntityExtractor:
    """
    Extracts (Subject, Predicate, Object) triples from text chunks
    using an LLM (via OllamaLLM).
    """

    def __init__(self, llm):
        """
        Args:
            llm: An OllamaLLM instance for generating extraction responses.
        """
        self.llm = llm

    def extract_triples(self, text_chunk: str) -> list[tuple]:
        """
        Extract triples from a single text chunk.

        Args:
            text_chunk: The text to extract relationships from.

        Returns:
            List of (subject, predicate, object) tuples with normalized entities.
        """
        if not text_chunk or not text_chunk.strip():
            return []

        # Truncate very long chunks to avoid LLM context overflow
        max_chars = 2000
        if len(text_chunk) > max_chars:
            text_chunk = text_chunk[:max_chars]

        prompt = EXTRACTION_PROMPT.format(text=text_chunk)

        try:
            response = self.llm.generate_response(
                prompt=prompt,
                max_tokens=500,
                temperature=0.1  # Low temperature for structured output
            )

            if not response:
                return []

            raw_triples = _parse_triples_response(response)

            # Normalize entities
            normalized = []
            for s, p, o in raw_triples:
                ns = _normalize_entity(s)
                no = _normalize_entity(o)
                np_ = p.strip().lower().replace(' ', '_')

                # Filter out empty or very short entities
                if len(ns) < 2 or len(no) < 2 or len(np_) < 2:
                    continue

                normalized.append((ns, np_, no))

            return normalized

        except Exception as e:
            logger.error(f"[EntityExtractor] Failed to extract triples: {e}")
            return []

    def extract_from_chunks(self, chunks: list[dict]) -> tuple[list[tuple], list[dict]]:
        """
        Extract triples from a batch of text chunks.

        Args:
            chunks: List of chunk dicts, each with at least {"text": str}
                    and optional metadata like {"page": int, "source": str}

        Returns:
            Tuple of:
            - all_triples: List of (subject, predicate, object) tuples
            - triple_sources: Parallel list of source metadata dicts for each triple
        """
        all_triples = []
        triple_sources = []

        total = len(chunks)
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            if not text.strip():
                continue

            logger.info(f"[EntityExtractor] Processing chunk {i + 1}/{total}...")

            triples = self.extract_triples(text)

            for triple in triples:
                all_triples.append(triple)
                triple_sources.append({
                    "page": chunk.get("page", "N/A"),
                    "source": chunk.get("source", "Unknown"),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "chunk_text_preview": text[:100],
                })

        # Deduplicate identical triples (keep first occurrence)
        seen = set()
        deduped_triples = []
        deduped_sources = []
        for triple, source in zip(all_triples, triple_sources):
            key = triple  # (s, p, o) tuple is hashable
            if key not in seen:
                seen.add(key)
                deduped_triples.append(triple)
                deduped_sources.append(source)

        logger.info(f"[EntityExtractor] Extracted {len(deduped_triples)} unique triples "
                     f"from {total} chunks (before dedup: {len(all_triples)})")

        return deduped_triples, deduped_sources
