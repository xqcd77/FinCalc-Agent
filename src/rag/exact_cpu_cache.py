"""Bounded, thread-safe CPU memoization with value-copy isolation.

Keys must contain all semantic inputs. Disable for paired baseline checks via
FINCALC_EXACT_CPU_CACHE=0. No model replies or evaluation labels are cached here.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import os
import threading


def signature(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


class ExactCPUCache:
    def __init__(self, capacity=512):
        self.capacity = capacity
        self.values = OrderedDict()
        self.lock = threading.RLock()
        self.hits = self.misses = 0

    def get_or_compute(self, key, compute):
        if os.environ.get('FINCALC_EXACT_CPU_CACHE', '1') == '0':
            return compute()
        # Hold the lock across compute: duplicate concurrent requests are scored
        # once, and cached dictionaries never escape by reference.
        with self.lock:
            if key in self.values:
                self.hits += 1
                self.values.move_to_end(key)
                return deepcopy(self.values[key])
            self.misses += 1
            value = compute()
            self.values[key] = deepcopy(value)
            if len(self.values) > self.capacity:
                self.values.popitem(last=False)
            return value

    def stats(self):
        with self.lock:
            return dict(hits=self.hits, misses=self.misses, entries=len(self.values))
