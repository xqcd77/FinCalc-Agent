"""Bounded reuse of identical greedy requests for a fixed loaded model.

No approximate matching, prompt rewriting, token truncation or KV reuse.
Only complete JSON-object responses are eligible; errors are never cached.
"""
from collections import OrderedDict
from copy import deepcopy
import json


class ExactInferenceCache:
    def __init__(self, capacity=64):
        self.capacity = capacity
        self.items = OrderedDict()
        self.hits = self.misses = 0

    @staticmethod
    def key(payload):
        # Unknown options remain in the key. Only the diagnostic bypass is excluded.
        return json.dumps({k: v for k, v in payload.items()
                           if k != '_bypass_exact_cache'}, ensure_ascii=False,
                          sort_keys=True, separators=(',', ':'))

    def get(self, payload):
        if payload.get('_bypass_exact_cache'):
            return None
        key = self.key(payload)
        if key not in self.items:
            self.misses += 1
            return None
        self.hits += 1
        self.items.move_to_end(key)
        return deepcopy(self.items[key])

    def put(self, payload, result):
        try:
            content = result['choices'][0]['message']['content']
            if not isinstance(json.loads(content), dict):
                return
        except (KeyError, TypeError, json.JSONDecodeError):
            return
        key = self.key(payload)
        self.items[key] = deepcopy(result)
        self.items.move_to_end(key)
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)
