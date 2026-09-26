"""Streaming candidate retrieval and evaluation for the Phase 3 benchmark."""

import hashlib
import heapq
import re
import time
import os
from collections import Counter, defaultdict
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None
try:
    import resource
except ImportError:
    resource = None


def process_rss_bytes():
    if psutil:
        return psutil.Process().memory_info().rss
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_fault_count", wintypes.DWORD),
                        ("peak_working_set_size", ctypes.c_size_t),
                        ("working_set_size", ctypes.c_size_t),
                        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                        ("quota_paged_pool_usage", ctypes.c_size_t),
                        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                        ("quota_non_paged_pool_usage", ctypes.c_size_t),
                        ("pagefile_usage", ctypes.c_size_t),
                        ("peak_pagefile_usage", ctypes.c_size_t)]

        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        get_memory_info = psapi.GetProcessMemoryInfo
        get_memory_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
        get_memory_info.restype = wintypes.BOOL
        get_memory_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters.working_set_size
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 if resource else 0

from business_entity_resolution.workspace import iter_tsv


def parse_match_ids(value):
    return {item.strip() for item in value.split(",") if item.strip()}


def normalize(value):
    return re.sub(r"[^0-9a-z]+", " ", value.casefold()).strip()


def route_keys(row):
    name = normalize(row["business_name"])
    address = normalize(row["business_address"])
    country = row["country"].strip().casefold()
    return {
        "name": (country, name) if name else None,
        "address": (country, address) if address else None,
        "name_prefix": (country, name[:8]) if name else None,
    }


def _sample_rank(entity_id):
    return int(hashlib.sha256(entity_id.encode("utf-8")).hexdigest()[:16], 16)


def select_development_sample(source1_path, truth_path, sample_size):
    """Select a deterministic, match-count-stratified sample without row alignment."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    buckets = defaultdict(list)
    per_bucket = max(1, sample_size // 6)
    selected_truth = {}
    for truth in iter_tsv(truth_path):
        reference_id = truth["source1_entity_id"]
        match_ids = parse_match_ids(truth["matched_entity_ids"])
        bucket = min(len(match_ids), 5)
        ranked = (_sample_rank(reference_id), reference_id, match_ids)
        heap = buckets[bucket]
        heapq.heappush(heap, (-ranked[0], ranked[1], ranked[2]))
        if len(heap) > per_bucket:
            heapq.heappop(heap)
    for rows in buckets.values():
        for _, reference_id, match_ids in rows:
            selected_truth[reference_id] = match_ids
    selected = []
    for source in iter_tsv(source1_path):
        reference_id = source["entity_id"]
        if reference_id in selected_truth:
            selected.append((source, selected_truth[reference_id]))
    selected.sort(key=lambda item: item[0]["entity_id"])
    return selected[:sample_size]


def _sample_queries(sample):
    queries = {route: defaultdict(set) for route in ("name", "address", "name_prefix")}
    for source, _ in sample:
        for route, key in route_keys(source).items():
            if key is not None:
                queries[route][key].add(source["entity_id"])
    return queries


def retrieve_routes(sample, target_paths):
    """Stream targets and return candidates for each route and reference."""
    queries = _sample_queries(sample)
    candidates = {
        route: defaultdict(set)
        for route in ("name", "address", "name_prefix")
    }
    for target_path in target_paths:
        for row in iter_tsv(target_path):
            for route, key in route_keys(row).items():
                if key is None:
                    continue
                for reference_id in queries[route].get(key, ()):
                    candidates[route][reference_id].add(row["entity_id"])
    return candidates


def f05(truth, prediction):
    true_positive = len(truth & prediction)
    false_positive = len(prediction - truth)
    false_negative = len(truth - prediction)
    denominator = 1.25 * true_positive + false_positive + 0.25 * false_negative
    return 1.0 if denominator == 0 else 1.25 * true_positive / denominator


def evaluate(sample, candidates):
    truth_by_id = {source["entity_id"]: truth for source, truth in sample}
    routes = list(candidates)
    union = {
        reference_id: set().union(*(candidates[route].get(reference_id, set()) for route in routes))
        for reference_id in truth_by_id
    }
    result = {}
    for route in routes + ["union"]:
        selected = union if route == "union" else candidates[route]
        recalls = [bool(truth_by_id[reference_id] <= selected.get(reference_id, set()))
                   for reference_id in truth_by_id if truth_by_id[reference_id]]
        scores = [f05(truth_by_id[reference_id], selected.get(reference_id, set()))
                  for reference_id in truth_by_id]
        sizes = [len(selected.get(reference_id, set())) for reference_id in truth_by_id]
        result[route] = {
            "candidate_recall": sum(recalls) / len(recalls) if recalls else 1.0,
            "oracle_macro_f05": sum(scores) / len(scores),
            "candidate_count_min": min(sizes),
            "candidate_count_mean": sum(sizes) / len(sizes),
            "candidate_count_max": max(sizes),
        }
    return result


def run_benchmark(source1_path, truth_path, target_paths, sample_size):
    start = time.perf_counter()
    sample = select_development_sample(source1_path, truth_path, sample_size)
    candidates = retrieve_routes(sample, target_paths)
    metrics = evaluate(sample, candidates)
    peak_rss = process_rss_bytes()
    return {
        "sample_size": len(sample),
        "sample_countries": dict(Counter(source["country"] for source, _ in sample)),
        "target_files": [str(path) for path in target_paths],
        "routes": metrics,
        "runtime_seconds": time.perf_counter() - start,
        "peak_ram_gib": peak_rss / (1024 ** 3),
        "peak_ram_method": "process RSS via psutil/Windows API/ru_maxrss",
    }