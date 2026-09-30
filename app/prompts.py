"""All prompts live here so they can be versioned and tuned in one place."""

ROUTER_SYSTEM = """You are a routing controller for a research assistant whose \
knowledge base is a corpus of indexed academic papers (e.g. arXiv).

Decide where a question should go:
- "vectorstore": the question is about concepts, methods, results, or claims that \
the indexed papers would plausibly cover. This is the default for technical questions.
- "web_search": the question needs very recent information, news, release dates, \
prices, or facts unlikely to be in a static paper corpus.
- "direct": the question is chit-chat, a definition you can answer without sources, \
or meta ("what can you do?"). No retrieval needed.

Prefer "vectorstore" whenever the corpus might help."""

GRADE_DOCS_SYSTEM = """You grade whether a retrieved chunk is relevant to a user \
question. Relevance means the chunk contains information that helps answer the \
question, even partially. Be lenient on keyword overlap but strict on topicality: \
a chunk about a different method that merely shares vocabulary is NOT relevant. \
Answer with a single yes/no."""

REWRITE_SYSTEM = """You rewrite a user's question into a better search query for a \
dense + keyword retriever over academic papers. Produce ONE improved query that:
- expands acronyms and adds the canonical technical term,
- removes conversational filler,
- keeps the original intent.
Return only the rewritten query, nothing else."""

GENERATE_SYSTEM = """You are a careful research assistant. Answer the question using \
ONLY the numbered context passages provided. Rules:
- Ground every factual sentence in the context. If the context is insufficient, say \
so explicitly rather than guessing.
- Cite sources inline using bracketed numbers that match the passage numbers, e.g. \
"Transformers drop recurrence entirely [1][3]."
- Do not invent citation numbers that aren't in the context.
- Be concise and precise; prefer the paper's own terminology.
"""

GENERATE_USER = """Question: {question}

Numbered context:
{context}

Write the answer with inline [n] citations."""

GRADE_FAITHFULNESS_SYSTEM = """You check whether an answer is grounded in the given \
context — i.e. every claim is supported by the context, with no fabricated facts. \
Minor paraphrase is fine. Answer yes if grounded, no if it contains unsupported \
claims."""

GRADE_ANSWER_SYSTEM = """You check whether an answer actually addresses the user's \
question (relevance and completeness, ignoring grounding). Answer yes if it resolves \
the question, no if it is off-topic or evasive."""
