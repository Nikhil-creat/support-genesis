"""Hybrid retrieval tool: dense embedding search + BM25 sparse search,
fused and optionally re-ranked with a cross-encoder.

This module defines the retrieval algorithm independent of any specific
vector store so it can back either an in-process index (tests, small corpora)
or a production pgvector/Redis-backed store, by swapping the `VectorIndex`
and `SparseIndex` implementations.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from app.schemas import KnowledgeBaseQuery, KnowledgeBaseResult, KnowledgeChunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class IndexedDocument:
    chunk_id: str
    text: str
    source: str
    embedding: np.ndarray


class BM25SparseIndex:
    """Minimal, dependency-free BM25 implementation (Okapi BM25)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: dict[str, list[str]] = {}
        self._doc_freq: Counter = Counter()
        self._avgdl = 0.0

    def add(self, chunk_id: str, text: str) -> None:
        tokens = _tokenize(text)
        self._docs[chunk_id] = tokens
        for term in set(tokens):
            self._doc_freq[term] += 1
        self._avgdl = sum(len(t) for t in self._docs.values()) / max(len(self._docs), 1)

    def score(self, query: str) -> dict[str, float]:
        n = len(self._docs)
        if n == 0:
            return {}
        q_terms = _tokenize(query)
        scores: dict[str, float] = defaultdict(float)
        for chunk_id, tokens in self._docs.items():
            doc_len = len(tokens)
            tf = Counter(tokens)
            for term in q_terms:
                if term not in tf:
                    continue
                df = self._doc_freq[term]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                numerator = tf[term] * (self.k1 + 1)
                denominator = tf[term] + self.k1 * (1 - self.b + self.b * doc_len / max(self._avgdl, 1e-6))
                scores[chunk_id] += idf * (numerator / denominator)
        if scores:
            max_score = max(scores.values()) or 1.0
            scores = {k: v / max_score for k, v in scores.items()}
        return scores


class DenseVectorIndex:
    """Cosine-similarity dense index over pre-computed embeddings (in-memory)."""

    def __init__(self) -> None:
        self._documents: list[IndexedDocument] = []

    def add(self, doc: IndexedDocument) -> None:
        self._documents.append(doc)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
        return float(np.dot(a, b) / denom)

    def score(self, query_embedding: np.ndarray) -> dict[str, float]:
        scores = {doc.chunk_id: self._cosine(query_embedding, doc.embedding) for doc in self._documents}
        if scores:
            max_score = max(scores.values()) or 1.0
            scores = {k: max(v, 0.0) / max_score for k, v in scores.items()}
        return scores

    def get(self, chunk_id: str) -> IndexedDocument | None:
        return next((d for d in self._documents if d.chunk_id == chunk_id), None)


class CrossEncoderReranker:
    """Lightweight lexical-overlap cross-encoder stand-in.

    In production this delegates to a real cross-encoder (e.g.
    `cross-encoder/ms-marco-MiniLM-L-6-v2`) scoring (query, passage) pairs
    directly; the interface below is what that integration point looks like.
    """

    def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        q_tokens = set(_tokenize(query))
        scores = []
        for passage in passages:
            p_tokens = set(_tokenize(passage))
            overlap = len(q_tokens & p_tokens)
            scores.append(overlap / max(len(q_tokens), 1))
        return scores


class HybridKnowledgeBase:
    def __init__(self, embed_fn) -> None:
        """`embed_fn` is an async callable: str -> np.ndarray (production: calls the embedding API)."""
        self._dense = DenseVectorIndex()
        self._sparse = BM25SparseIndex()
        self._reranker = CrossEncoderReranker()
        self._embed_fn = embed_fn

    async def ingest(self, chunk_id: str, text: str, source: str) -> None:
        embedding = await self._embed_fn(text)
        self._dense.add(IndexedDocument(chunk_id=chunk_id, text=text, source=source, embedding=embedding))
        self._sparse.add(chunk_id, text)

    async def query(self, request: KnowledgeBaseQuery) -> KnowledgeBaseResult:
        start = time.perf_counter()
        query_embedding = await self._embed_fn(request.query)
        dense_scores = self._dense.score(query_embedding)
        sparse_scores = self._sparse.score(request.query)

        all_ids = set(dense_scores) | set(sparse_scores)
        fused = {
            cid: request.dense_weight * dense_scores.get(cid, 0.0)
            + request.sparse_weight * sparse_scores.get(cid, 0.0)
            for cid in all_ids
        }
        ranked_ids = sorted(fused, key=fused.get, reverse=True)[: max(request.top_k * 3, request.top_k)]

        candidates = [self._dense.get(cid) for cid in ranked_ids]
        candidates = [c for c in candidates if c is not None]

        rerank_scores: dict[str, float] = {}
        if request.use_reranker and candidates:
            passage_scores = self._reranker.score_pairs(request.query, [c.text for c in candidates])
            rerank_scores = {c.chunk_id: s for c, s in zip(candidates, passage_scores)}

        chunks: list[KnowledgeChunk] = []
        for doc in candidates[: request.top_k]:
            final = rerank_scores.get(doc.chunk_id, fused.get(doc.chunk_id, 0.0))
            chunks.append(KnowledgeChunk(
                chunk_id=doc.chunk_id,
                text=doc.text,
                source=doc.source,
                dense_score=dense_scores.get(doc.chunk_id, 0.0),
                sparse_score=sparse_scores.get(doc.chunk_id, 0.0),
                rerank_score=rerank_scores.get(doc.chunk_id),
                final_score=final,
            ))
        chunks.sort(key=lambda c: c.final_score, reverse=True)

        latency_ms = (time.perf_counter() - start) * 1000
        return KnowledgeBaseResult(chunks=chunks, query_latency_ms=latency_ms)


async def query_knowledge_base(kb: HybridKnowledgeBase, request: KnowledgeBaseQuery) -> KnowledgeBaseResult:
    """Tool entrypoint bound into the MCP registry / LangGraph tool node."""
    return await kb.query(request)
