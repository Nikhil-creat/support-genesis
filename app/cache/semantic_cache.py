"""Semantic vector cache: intercepts semantically-identical queries and
returns previously computed agent trajectories, skipping the LLM/graph run.

Uses Redis as the backing store: each entry is stored as a hash containing
the serialized embedding (as bytes), the original query, and the cached
trajectory (JSON). Similarity search is brute-force cosine over the
candidate set fetched by an approximate key-prefix scan, which is adequate
for moderate cache sizes; swap in RediSearch's native vector index (HNSW)
for large-scale deployments without changing this class's public interface.
"""
from __future__ import annotations

import json
import struct
import time
from typing import Any

import numpy as np
import redis.asyncio as aioredis

from app.config import get_settings

_CACHE_KEY_PREFIX = "semcache:"


def _serialize_vector(vec: np.ndarray) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec.astype(np.float32).tolist())


def _deserialize_vector(raw: bytes) -> np.ndarray:
    n = len(raw) // 4
    return np.array(struct.unpack(f"{n}f", raw), dtype=np.float32)


class SemanticCache:
    def __init__(self, embed_fn, redis_client: aioredis.Redis | None = None) -> None:
        settings = get_settings()
        self._embed_fn = embed_fn
        self._redis = redis_client or aioredis.from_url(settings.redis_url, decode_responses=False)
        self._threshold = settings.semantic_cache_similarity_threshold
        self._ttl = settings.semantic_cache_ttl_seconds

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
        return float(np.dot(a, b) / denom)

    async def lookup(self, query: str, max_candidates: int = 200) -> dict[str, Any] | None:
        query_vec = await self._embed_fn(query)
        best_score = -1.0
        best_entry: dict[str, Any] | None = None

        cursor = 0
        scanned = 0
        while True:
            cursor, keys = await self._redis.scan(cursor=cursor, match=f"{_CACHE_KEY_PREFIX}*", count=100)
            for key in keys:
                raw = await self._redis.hgetall(key)
                if not raw:
                    continue
                vec = _deserialize_vector(raw[b"embedding"])
                score = self._cosine(query_vec, vec)
                if score > best_score:
                    best_score = score
                    best_entry = {
                        "query": raw[b"query"].decode("utf-8"),
                        "trajectory": json.loads(raw[b"trajectory"]),
                        "similarity": score,
                    }
                scanned += 1
            if cursor == 0 or scanned >= max_candidates:
                break

        if best_entry is not None and best_score >= self._threshold:
            return best_entry
        return None

    async def store(self, query: str, trajectory: dict[str, Any]) -> None:
        query_vec = await self._embed_fn(query)
        key = f"{_CACHE_KEY_PREFIX}{int(time.time() * 1000)}:{hash(query) & 0xFFFFFFFF}"
        await self._redis.hset(key, mapping={
            "embedding": _serialize_vector(query_vec),
            "query": query.encode("utf-8"),
            "trajectory": json.dumps(trajectory).encode("utf-8"),
        })
        await self._redis.expire(key, self._ttl)

    async def close(self) -> None:
        await self._redis.aclose()
