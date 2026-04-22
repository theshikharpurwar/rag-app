# Benchmark datasets

## Schema (`*.yaml`)

```yaml
meta:
  name: <string>              # dataset id for reports
  pdf_id: <string>            # stored in Qdrant payload + BM25/KG files under INDICES_DIR
  fixture_pdf: <filename>     # path relative to this directory
  collection_name: documents  # optional; Qdrant collection (default: documents)
  description: <string>

questions:
  - id: <string>
    type: keyword | multi_hop | numerical | broad
    question: <string>
    gold_answer: <string>     # reference for LLM-judge (context_recall) and conciseness
    gold_pages: [1, 2]        # optional 1-based page hints; used for page-overlap in context_recall
```

## Adding a new dataset

1. Place a PDF in this directory (or reference a path under `fixture_pdf` relative to this dir).
2. Run ingestion once:  
   `cd python && python -m evaluation.run_benchmark --dataset evaluation/datasets/your.yaml`  
   (or `--skip-ingest` after the first successful ingest).
3. Add enough questions to stress vector-only vs hybrid + agent (keywords, cross-page, numbers).

## Bundled `sample.yaml`

Paired with [fixture.pdf](fixture.pdf) (five-page Acme Corp handout). The fixed `pdf_id` is `eval_fixture_acme` so re-runs do not clash with app uploads.
