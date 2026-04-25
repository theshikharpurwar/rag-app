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


def _extract_triples_from_parsed(parsed) -> list[tuple]:
    """
    Recursively extract (subject, predicate, object) tuples from any nested
    JSON structure the LLM might return:
      - [[\"A\",\"B\",\"C\"]] or [{\"subject\":...}] (standard)
      - [[{\"subject\":...}]] or [[[\"A\",\"B\",\"C\"]]] (extra nesting from small models)
    """
    triples = []

    if not isinstance(parsed, list):
        return triples

    for item in parsed:
        if isinstance(item, dict):
            # Direct dict: {"subject": ..., "predicate": ..., "object": ...}
            s = item.get('subject', item.get('s', ''))
            p = item.get('predicate', item.get('p', item.get('relation', '')))
            o = item.get('object', item.get('o', ''))
            if s and p and o:
                triples.append((str(s), str(p), str(o)))
        elif isinstance(item, (list, tuple)):
            if len(item) >= 3 and all(isinstance(x, str) for x in item[:3]):
                # Flat string triple: ["A", "B", "C"]
                triples.append((str(item[0]), str(item[1]), str(item[2])))
            else:
                # Nested: recurse to unwrap [[{...}]] or [[["A","B","C"]]]
                triples.extend(_extract_triples_from_parsed(item))

    return triples


def _parse_triples_response(response: str) -> list[tuple]:
    """
    Parse LLM response into a list of (subject, predicate, object) tuples.
    Handles any nesting depth, markdown fences, and truncated JSON.
    """
    if not response:
        return []

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"```(?:json)?\s*", "", response).strip()
    text = text.replace("```", "").strip()

    # Try direct JSON parse first
    try:
        parsed = json.loads(text)
        result = _extract_triples_from_parsed(parsed)
        if result:
            return result
    except json.JSONDecodeError:
        pass

    # Fall back: locate the outermost JSON array via regex
    json_match = re.search(r'\[[\s\S]*\]', text)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            result = _extract_triples_from_parsed(parsed)
            if result:
                return result
        except json.JSONDecodeError:
            pass

    # Last resort: extract individual ["s","p","o"] triples via regex
    # (handles truncated JSON where the outer array is broken)
    triple_pattern = re.compile(
        r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]'
    )
    matches = triple_pattern.findall(text)
    if matches:
        return [(s.strip(), p.strip(), o.strip()) for s, p, o in matches]

    return []


class EntityExtractor:
    """
    Extracts (Subject, Predicate, Object) triples from text chunks
    using an LLM (via OllamaLLM).

    Optimized: batches multiple chunks per LLM call to reduce round-trips.
    """

    def __init__(self, llm, batch_size: int = 3):
        """
        Args:
            llm: An OllamaLLM instance for generating extraction responses.
            batch_size: Number of chunks to send per LLM call (default 3).
        """
        self.llm = llm
        self.batch_size = batch_size

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
                temperature=0.1,  # Low temperature for structured output
                response_format="json"
            )

            if not response:
                return []

            raw_triples = _parse_triples_response(response)
            if not raw_triples:
                logger.debug(f"[EntityExtractor] 0 triples parsed. Raw response (first 300 chars): {response[:300]!r}")
            return _normalize_triples(raw_triples)

        except Exception as e:
            logger.error(f"[EntityExtractor] Failed to extract triples: {e}")
            return []

    def _extract_batch(self, texts: list[str]) -> list[tuple]:
        """
        Extract triples from multiple text chunks in a single LLM call.

        Args:
            texts: List of text strings to process together.

        Returns:
            List of (subject, predicate, object) tuples from all texts.
        """
        # Build a combined prompt with labeled chunks
        combined_text = ""
        for i, text in enumerate(texts):
            # Truncate each chunk
            truncated = text[:1500] if len(text) > 1500 else text
            combined_text += f"--- CHUNK {i+1} ---\n{truncated}\n\n"

        prompt = f"""Extract entity relationships from the text below.
Output a JSON array. Each item must be exactly ["subject", "predicate", "object"].

Example output:
[["Alice","works_at","Google"],["Google","is_a","Company"],["Alice","knows","Python"]]

Rules:
- subject and object: short noun phrases only (person, technology, place, concept)
- predicate: a short verb phrase (works_at, uses, is_a, part_of, studied_at, created)
- maximum 15 triples total
- return [] if no clear relationships exist
- no explanation, no markdown, just the JSON array

Text:
{combined_text}
JSON:"""

        try:
            response = self.llm.generate_response(
                prompt=prompt,
                max_tokens=500,
                temperature=0.1
            )

            if not response:
                return []

            raw_triples = _parse_triples_response(response)
            logger.info(f"[EntityExtractor] Raw parse: {len(raw_triples)} triples from response ({len(response)} chars)")
            if not raw_triples:
                logger.info(f"[EntityExtractor] Batch 0 triples parsed. Raw response (first 500 chars): {response[:500]!r}")
            normalized = _normalize_triples(raw_triples)
            logger.info(f"[EntityExtractor] After normalize: {len(normalized)} triples")
            return normalized

        except Exception as e:
            logger.error(f"[EntityExtractor] Batch extraction failed: {e}")
            return []

    def extract_from_chunks(self, chunks: list[dict]) -> tuple[list[tuple], list[dict]]:
        """
        Extract triples from a batch of text chunks using batched LLM calls.

        Processes chunks in groups of `batch_size` to minimize Ollama round-trips.

        Args:
            chunks: List of chunk dicts, each with at least {"text": str}
                    and optional metadata like {"page": int, "source": str}

        Returns:
            Tuple of:
            - all_triples: List of (subject, predicate, object) tuples
            - triple_sources: Parallel list of source metadata dicts for each triple
        """
        import time

        all_triples = []
        triple_sources = []

        # Filter out empty chunks
        valid_chunks = [c for c in chunks if c.get("text", "").strip()]
        total = len(valid_chunks)

        if total == 0:
            return [], []

        num_batches = (total + self.batch_size - 1) // self.batch_size
        start_time = time.time()

        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, total)
            batch_chunks = valid_chunks[batch_start:batch_end]
            batch_time = time.time()

            # Extract triples from the batch
            texts = [c.get("text", "") for c in batch_chunks]
            triples = self._extract_batch(texts)

            elapsed = time.time() - batch_time
            logger.info(f"[EntityExtractor] Batch {batch_idx + 1}/{num_batches} "
                         f"({batch_end}/{total} chunks): {len(triples)} triples in {elapsed:.1f}s")

            # Assign source metadata from the first chunk in batch
            # (triples span multiple chunks, so we attribute to the batch's page range)
            for triple in triples:
                all_triples.append(triple)
                # Find the best source attribution by matching entity text
                best_source = batch_chunks[0]  # default to first chunk
                triple_text = f"{triple[0]} {triple[2]}".lower()
                for chunk in batch_chunks:
                    if any(word in chunk.get("text", "").lower() for word in triple_text.split()[:2]):
                        best_source = chunk
                        break

                triple_sources.append({
                    "page": best_source.get("page", "N/A"),
                    "source": best_source.get("source", "Unknown"),
                    "chunk_index": best_source.get("chunk_index", 0),
                    "chunk_text": best_source.get("text", ""),
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

        total_time = time.time() - start_time
        logger.info(f"[EntityExtractor] Extracted {len(deduped_triples)} unique triples "
                     f"from {total} chunks in {total_time:.1f}s "
                     f"(before dedup: {len(all_triples)}, "
                     f"batches: {num_batches}, batch_size: {self.batch_size})")

        return deduped_triples, deduped_sources


def _normalize_triples(raw_triples: list[tuple]) -> list[tuple]:
    """Normalize a list of raw triples — lowercase, strip articles, filter short."""
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

