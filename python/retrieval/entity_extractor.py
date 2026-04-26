# python/retrieval/entity_extractor.py
"""
Knowledge graph extraction strategies.

Supports:
- LLM triple extraction with batching, optional concurrency, and audit metadata
- Regex-based noun phrase extraction with deterministic co-occurrence edges
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

logger = logging.getLogger(__name__)


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

_CAPITALIZED_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][a-z0-9]+(?:[-/][A-Z][a-z0-9]+)?)(?:\s+(?:[A-Z][a-z0-9]+(?:[-/][A-Z][a-z0-9]+)?)){1,5}\b"
)
_TECH_IDENTIFIER_RE = re.compile(
    r"\b(?:[a-z]+(?:_[a-z0-9]+)+|[a-z]+[A-Z][A-Za-z0-9]*)\b"
)
_ACRONYM_RE = re.compile(
    r"\b(?:[A-Z]{2,}(?:\d+)?|[A-Z](?:&|/|-)[A-Z](?:[A-Z])*)\b"
)


def _normalize_entity(entity: str) -> str:
    entity = entity.strip().lower()
    entity = re.sub(r"^(the|a|an)\s+", "", entity)
    entity = entity.rstrip(".,;:!?")
    entity = re.sub(r"\s+", " ", entity)
    return entity.strip()


def _extract_triples_from_parsed(parsed) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    if not isinstance(parsed, list):
        return triples

    for item in parsed:
        if isinstance(item, dict):
            s = item.get("subject", item.get("s", ""))
            p = item.get("predicate", item.get("p", item.get("relation", "")))
            o = item.get("object", item.get("o", ""))
            if s and p and o:
                triples.append((str(s), str(p), str(o)))
        elif isinstance(item, (list, tuple)):
            if len(item) >= 3 and all(isinstance(x, str) for x in item[:3]):
                triples.append((str(item[0]), str(item[1]), str(item[2])))
            else:
                triples.extend(_extract_triples_from_parsed(item))
    return triples


def _parse_triples_response(response: str) -> list[tuple[str, str, str]]:
    if not response:
        return []

    text = re.sub(r"```(?:json)?\s*", "", response).strip()
    text = text.replace("```", "").strip()

    try:
        parsed = json.loads(text)
        result = _extract_triples_from_parsed(parsed)
        if result:
            return result
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\[[\s\S]*\]", text)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            result = _extract_triples_from_parsed(parsed)
            if result:
                return result
        except json.JSONDecodeError:
            pass

    triple_pattern = re.compile(
        r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]'
    )
    matches = triple_pattern.findall(text)
    if matches:
        return [(s.strip(), p.strip(), o.strip()) for s, p, o in matches]

    return []


def _normalize_triples(raw_triples: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    normalized: list[tuple[str, str, str]] = []
    for s, p, o in raw_triples:
        ns = _normalize_entity(s)
        no = _normalize_entity(o)
        np_ = p.strip().lower().replace(" ", "_")
        if len(ns) < 2 or len(no) < 2 or len(np_) < 2:
            continue
        normalized.append((ns, np_, no))
    return normalized


def _chunk_preview(text: str, limit: int = 200) -> str:
    if not text:
        return ""
    trimmed = re.sub(r"\s+", " ", text.strip())
    return trimmed[:limit]


class EntityExtractor:
    """
    LLM triple extractor with audit instrumentation.
    """

    mode = "llm_triples"

    def __init__(
        self,
        llm,
        batch_size: int = 3,
        max_chars: int = 1200,
        concurrency: int = 1,
    ):
        self.llm = llm
        self.batch_size = max(1, int(batch_size))
        self.max_chars = max(100, int(max_chars))
        self.concurrency = max(1, int(concurrency))

    def _prepare_text(self, text: str) -> tuple[str, bool]:
        if not text:
            return "", False
        if len(text) > self.max_chars:
            return text[: self.max_chars], True
        return text, False

    def extract_triples(self, text_chunk: str) -> list[tuple[str, str, str]]:
        if not text_chunk or not text_chunk.strip():
            return []
        prepared, _ = self._prepare_text(text_chunk)
        prompt = EXTRACTION_PROMPT.format(text=prepared)
        try:
            response = self.llm.generate_response(
                prompt=prompt,
                max_tokens=500,
                temperature=0.1,
                response_format="json",
            )
            if not response:
                return []
            raw_triples = _parse_triples_response(response)
            if not raw_triples:
                logger.debug(
                    "[EntityExtractor] 0 triples parsed. Raw response (first 300 chars): %r",
                    response[:300],
                )
            return _normalize_triples(raw_triples)
        except Exception as e:
            logger.error("[EntityExtractor] Failed to extract triples: %s", e)
            return []

    def _extract_batch(
        self,
        batch_idx: int,
        batch_chunks: list[dict],
    ) -> dict:
        combined_text_parts: list[str] = []
        truncation_count = 0
        for i, chunk in enumerate(batch_chunks):
            truncated, was_truncated = self._prepare_text(chunk.get("text", ""))
            truncation_count += int(was_truncated)
            combined_text_parts.append(f"--- CHUNK {i + 1} ---\n{truncated}\n")
        combined_text = "\n".join(combined_text_parts)

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

        start = time.time()
        response = ""
        raw_triples: list[tuple[str, str, str]] = []
        normalized: list[tuple[str, str, str]] = []
        parser_errors = 0
        error = None

        try:
            response = self.llm.generate_response(
                prompt=prompt,
                max_tokens=500,
                temperature=0.1,
            )
            if response:
                raw_triples = _parse_triples_response(response)
                if not raw_triples:
                    parser_errors += 1
                normalized = _normalize_triples(raw_triples)
            else:
                parser_errors += 1
        except Exception as exc:
            error = str(exc)
            parser_errors += 1
            logger.error("[EntityExtractor] Batch extraction failed: %s", exc)

        elapsed = time.time() - start
        logger.info(
            "[EntityExtractor] Batch %s: %s triples in %.1fs (truncated_chunks=%s)",
            batch_idx + 1,
            len(normalized),
            elapsed,
            truncation_count,
        )
        if truncation_count:
            logger.info(
                "[EntityExtractor] Batch %s truncated %s chunk(s) to %s chars",
                batch_idx + 1,
                truncation_count,
                self.max_chars,
            )

        return {
            "batch_idx": batch_idx,
            "elapsed_seconds": elapsed,
            "response": response,
            "raw_triples": raw_triples,
            "normalized_triples": normalized,
            "parser_errors": parser_errors,
            "truncation_count": truncation_count,
            "error": error,
            "chunk_range": [batch_chunks[0].get("chunk_index", 0), batch_chunks[-1].get("chunk_index", 0)],
            "chunk_previews": [_chunk_preview(c.get("text", "")) for c in batch_chunks],
        }

    def extract_from_chunks(
        self,
        chunks: list[dict],
    ) -> tuple[list[tuple[str, str, str]], list[dict], dict]:
        all_triples: list[tuple[str, str, str]] = []
        triple_sources: list[dict] = []
        valid_chunks = [c for c in chunks if c.get("text", "").strip()]
        total = len(valid_chunks)
        if total == 0:
            return [], [], self._empty_audit()

        num_batches = (total + self.batch_size - 1) // self.batch_size
        start_time = time.time()
        batches = [
            valid_chunks[i : i + self.batch_size]
            for i in range(0, total, self.batch_size)
        ]

        if self.concurrency > 1 and len(batches) > 1:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                batch_results = list(
                    pool.map(lambda args: self._extract_batch(*args), enumerate(batches))
                )
        else:
            batch_results = [self._extract_batch(idx, batch) for idx, batch in enumerate(batches)]

        batch_results.sort(key=lambda item: item["batch_idx"])

        total_parser_errors = 0
        total_truncations = 0
        before_dedup = 0

        for result, batch_chunks in zip(batch_results, batches):
            triples = result["normalized_triples"]
            raw_triples = result["raw_triples"]
            before_dedup += len(triples)
            total_parser_errors += int(result["parser_errors"])
            total_truncations += int(result["truncation_count"])

            for triple in triples:
                all_triples.append(triple)
                best_source = batch_chunks[0]
                triple_text = f"{triple[0]} {triple[2]}".lower()
                for chunk in batch_chunks:
                    lowered = chunk.get("text", "").lower()
                    if any(word in lowered for word in triple_text.split()[:2]):
                        best_source = chunk
                        break
                triple_sources.append(
                    {
                        "page": best_source.get("page", "N/A"),
                        "source": best_source.get("source", "Unknown"),
                        "chunk_index": best_source.get("chunk_index", 0),
                        "chunk_text": best_source.get("text", ""),
                        "chunk_text_preview": _chunk_preview(best_source.get("text", "")),
                    }
                )
            logger.info(
                "[EntityExtractor] Batch %s/%s (%s/%s chunks): %s triples in %.1fs",
                result["batch_idx"] + 1,
                num_batches,
                min((result["batch_idx"] + 1) * self.batch_size, total),
                total,
                len(raw_triples),
                result["elapsed_seconds"],
            )

        seen = set()
        deduped_triples: list[tuple[str, str, str]] = []
        deduped_sources: list[dict] = []
        for triple, source in zip(all_triples, triple_sources):
            if triple not in seen:
                seen.add(triple)
                deduped_triples.append(triple)
                deduped_sources.append(source)

        total_time = time.time() - start_time
        logger.info(
            "[EntityExtractor] Extracted %s unique triples from %s chunks in %.1fs "
            "(before dedup: %s, batches: %s, batch_size: %s, truncations: %s)",
            len(deduped_triples),
            total,
            total_time,
            len(all_triples),
            num_batches,
            self.batch_size,
            total_truncations,
        )

        audit = {
            "mode": self.mode,
            "chunks_processed": total,
            "batch_size": self.batch_size,
            "concurrency": self.concurrency,
            "max_chars": self.max_chars,
            "num_batches": num_batches,
            "elapsed_seconds": total_time,
            "triples_before_dedup": before_dedup,
            "triples_after_dedup": len(deduped_triples),
            "triples_per_chunk": (len(deduped_triples) / total) if total else 0.0,
            "parser_error_count": total_parser_errors,
            "truncation_count": total_truncations,
            "batches": batch_results,
        }
        return deduped_triples, deduped_sources, audit

    def _empty_audit(self) -> dict:
        return {
            "mode": self.mode,
            "chunks_processed": 0,
            "batch_size": self.batch_size,
            "concurrency": self.concurrency,
            "max_chars": self.max_chars,
            "num_batches": 0,
            "elapsed_seconds": 0.0,
            "triples_before_dedup": 0,
            "triples_after_dedup": 0,
            "triples_per_chunk": 0.0,
            "parser_error_count": 0,
            "truncation_count": 0,
            "batches": [],
        }


class NounPhraseCooccurrenceExtractor:
    """
    Deterministic extractor that creates co-occurrence edges from regex-matched entities.
    """

    mode = "noun_phrase_cooccurrence"

    def __init__(self, extractor_name: str = "regex"):
        self.extractor_name = extractor_name
        if extractor_name == "spacy":
            raise NotImplementedError(
                "KG_NP_EXTRACTOR=spacy is not implemented yet; use KG_NP_EXTRACTOR=regex."
            )
        if extractor_name != "regex":
            raise ValueError(f"Unsupported KG_NP_EXTRACTOR: {extractor_name}")

    def _extract_entities_regex(self, text: str) -> list[str]:
        matches = []
        for pattern in (_CAPITALIZED_PHRASE_RE, _TECH_IDENTIFIER_RE, _ACRONYM_RE):
            matches.extend(pattern.findall(text or ""))

        normalized = []
        seen = set()
        for match in matches:
            entity = _normalize_entity(match.replace("_", " "))
            if len(entity) < 2:
                continue
            if entity in seen:
                continue
            seen.add(entity)
            normalized.append(entity)
        return normalized

    def extract_from_chunks(
        self,
        chunks: list[dict],
    ) -> tuple[list[tuple[str, str, str]], list[dict], dict]:
        start = time.time()
        triples: list[tuple[str, str, str]] = []
        sources: list[dict] = []
        seen_triples = set()
        truncation_count = 0
        per_chunk = []
        valid_chunks = [c for c in chunks if c.get("text", "").strip()]

        for chunk in valid_chunks:
            entities = self._extract_entities_regex(chunk.get("text", ""))
            per_chunk.append(
                {
                    "chunk_index": chunk.get("chunk_index", 0),
                    "entity_count": len(entities),
                    "entities": entities[:20],
                    "chunk_preview": _chunk_preview(chunk.get("text", "")),
                }
            )
            if len(entities) < 2:
                continue
            for left, right in combinations(sorted(entities), 2):
                triple = (left, "co_occurs_with", right)
                if triple in seen_triples:
                    continue
                seen_triples.add(triple)
                triples.append(triple)
                sources.append(
                    {
                        "page": chunk.get("page", "N/A"),
                        "source": chunk.get("source", "Unknown"),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "chunk_text": chunk.get("text", ""),
                        "chunk_text_preview": _chunk_preview(chunk.get("text", "")),
                    }
                )

        elapsed = time.time() - start
        audit = {
            "mode": self.mode,
            "extractor": self.extractor_name,
            "chunks_processed": len(valid_chunks),
            "elapsed_seconds": elapsed,
            "triples_before_dedup": len(triples),
            "triples_after_dedup": len(triples),
            "triples_per_chunk": (len(triples) / len(valid_chunks)) if valid_chunks else 0.0,
            "parser_error_count": 0,
            "truncation_count": truncation_count,
            "chunks": per_chunk,
        }
        logger.info(
            "[CoOccurrenceExtractor] Extracted %s edges from %s chunks in %.2fs",
            len(triples),
            len(valid_chunks),
            elapsed,
        )
        return triples, sources, audit
