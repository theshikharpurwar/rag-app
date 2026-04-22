"""
Prompt templates for the Phase 3 agentic loop.

Kept in a dedicated module so prompt-engineering lives in one place,
separate from the orchestration logic.
"""

ROUTER_PROMPT = """You classify a user question into one of two retrieval strategies.

Return EXACTLY one token, no punctuation, no explanation:

- SPECIFIC — the question asks for a concrete fact, name, number, date, quotation,
  definition, or a single entity's attribute. BM25 keyword matching will help most.
- BROAD   — the question asks for a summary, comparison, relationship between
  concepts, multi-hop reasoning, or an overview. Graph traversal will help most.

Examples:
Q: "What year was the company founded?"          -> SPECIFIC
Q: "Who is the CEO?"                             -> SPECIFIC
Q: "How do the chapters relate to each other?"   -> BROAD
Q: "Summarize the author's main argument."       -> BROAD
Q: "What does section 3.2 say about latency?"    -> SPECIFIC
Q: "Compare the two proposed architectures."     -> BROAD

Question: {query}
Answer:"""


GRADER_PROMPT = """You grade how well an ANSWER is supported by a CONTEXT for a given QUERY.

Score two qualities, then combine:
- Faithfulness: does the answer stay within what the context supports (no hallucination)?
- Relevance:    does the answer actually address the query?

Return ONLY a single JSON object, nothing else, using this exact schema:
{{"score": <float between 0.0 and 1.0>, "reason": "<one short sentence>"}}

- 1.0 = fully grounded in context AND directly answers the query
- 0.5 = partially grounded or partially relevant
- 0.0 = hallucinated OR unrelated to the query

QUERY:
{query}

CONTEXT:
{context}

ANSWER:
{answer}

JSON:"""


REWRITE_PROMPT = """You rewrite a failed RAG question so the next retrieval finds better evidence.

The previous attempt produced a weak answer. The grader's reason is given.
Produce ONE reformulated question that:
- keeps the user's original intent,
- uses different keywords or a different angle,
- is a single sentence, no preamble, no quotes.

Original question: {query}
Grader feedback:   {reason}

Rewritten question:"""


DECOMPOSITION_PROMPT = """You break a complex question into at most {n} focused sub-questions for document retrieval.

Rules:
- Each sub-question must be self-contained and searchable (concrete nouns, no pronouns like "it" without antecedent).
- If the question is already simple or atomic, return it unchanged as a single line (no numbering).
- Otherwise return a numbered list only (one sub-question per line), no preamble, no explanation.

Question: {query}
Sub-questions:"""
