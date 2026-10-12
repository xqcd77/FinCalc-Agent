"""Exact-query ranking cache regression tests; no model or index files required."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "rag"))

import agent_v3


class QueryRankCacheTests(unittest.TestCase):
    def make_retriever(self, size=2):
        retriever = agent_v3.TargetedRetriever.__new__(agent_v3.TargetedRetriever)
        retriever.chunks = [{"chunk_id": "c0"}, {"chunk_id": "c1"}]
        retriever.bm25 = object()
        retriever.model = object()
        retriever.vectors = object()
        retriever._query_rank_cache = agent_v3.OrderedDict()
        retriever.query_rank_cache_hits = 0
        retriever.query_rank_cache_misses = 0
        return retriever

    def test_exact_query_reuses_identical_full_rankings(self):
        retriever = self.make_retriever()
        sparse = [(0, 2.0), (1, 1.0)]
        dense = [(1, 0.9), (0, 0.8)]
        with patch.object(agent_v3, "bm25_search", return_value=sparse) as bm25, \
             patch.object(agent_v3, "embed_queries", return_value=[[0.1, 0.2]]) as embed, \
             patch.object(agent_v3, "dense_search", return_value=dense) as dense_search:
            first = retriever._query_rankings("issuer year metric")
            second = retriever._query_rankings("issuer year metric")
            self.assertEqual(first, second)
            self.assertEqual(first, (tuple(sparse), tuple(dense)))
            self.assertEqual((bm25.call_count, embed.call_count, dense_search.call_count), (1, 1, 1))
            retriever._query_rankings("issuer year other metric")
            self.assertEqual((bm25.call_count, embed.call_count, dense_search.call_count), (2, 2, 2))
        self.assertEqual(retriever.query_rank_cache_hits, 1)
        self.assertEqual(retriever.query_rank_cache_misses, 2)

    def test_cache_is_bounded_and_evicts_least_recently_used_query(self):
        retriever = self.make_retriever()
        with patch.object(agent_v3, "QUERY_RANK_CACHE_SIZE", 2), \
             patch.object(agent_v3, "bm25_search", return_value=[(0, 1.0)]) as bm25, \
             patch.object(agent_v3, "embed_queries", return_value=[[0.1, 0.2]]), \
             patch.object(agent_v3, "dense_search", return_value=[(0, 1.0)]):
            retriever._query_rankings("A")
            retriever._query_rankings("B")
            retriever._query_rankings("A")  # A becomes most recently used.
            retriever._query_rankings("C")  # B is evicted.
            self.assertEqual(len(retriever._query_rank_cache), 2)
            retriever._query_rankings("B")
            self.assertEqual(bm25.call_count, 4)
        self.assertEqual(retriever.query_rank_cache_hits, 1)
        self.assertEqual(retriever.query_rank_cache_misses, 4)


if __name__ == "__main__":
    unittest.main()
