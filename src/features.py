"""Feature extraction for candidate entity pairs."""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import jellyfish
import numpy as np


FEATURE_NAMES = [
    "name_levenshtein",
    "name_jaccard",
    "name_tfidf_cosine",
    "addr_levenshtein",
    "addr_jaccard",
    "addr_tfidf_cosine",
    "common_name_tokens",
    "common_addr_tokens",
    "name_len_diff",
    "addr_len_diff",
    "exact_name_match",
    "exact_addr_match",
    "name_token_ratio",
    "addr_token_ratio",
]


def _char_ngrams(s: str, n: int = 3) -> list[str]:
    if not s:
        return []
    if len(s) < n:
        return [s]
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def _cosine_sim_ngrams(s1: str, s2: str, n: int = 3) -> float:
    if not s1 or not s2:
        return 0.0
    c1 = Counter(_char_ngrams(s1, n))
    c2 = Counter(_char_ngrams(s2, n))
    common = set(c1).intersection(c2)
    if not common:
        return 0.0
    dot = sum(c1[k] * c2[k] for k in common)
    norm1 = math.sqrt(sum(v * v for v in c1.values()))
    norm2 = math.sqrt(sum(v * v for v in c2.values()))
    return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0


def extract_pair_features(
    name1: str,
    addr1: str,
    name2: str,
    addr2: str,
) -> list[float]:
    """Compute similarity features between an S1 entity and an S2/S3 entity."""
    nl1, nl2 = len(name1), len(name2)
    max_nl = max(nl1, nl2, 1)
    lev_n = 1.0 - jellyfish.levenshtein_distance(name1, name2) / max_nl
    t1 = set(name1.split()) if name1 else set()
    t2 = set(name2.split()) if name2 else set()
    common_n = len(t1.intersection(t2))
    union_n = len(t1.union(t2))
    jacc_n = common_n / union_n if union_n > 0 else 0.0
    cos_n = _cosine_sim_ngrams(name1, name2, 3)

    al1, al2 = len(addr1), len(addr2)
    max_al = max(al1, al2, 1)
    lev_a = 1.0 - jellyfish.levenshtein_distance(addr1, addr2) / max_al
    at1 = set(addr1.split()) if addr1 else set()
    at2 = set(addr2.split()) if addr2 else set()
    common_a = len(at1.intersection(at2))
    union_a = len(at1.union(at2))
    jacc_a = common_a / union_a if union_a > 0 else 0.0
    cos_a = _cosine_sim_ngrams(addr1, addr2, 3)

    exact_n = 1.0 if name1 and name1 == name2 else 0.0
    exact_a = 1.0 if addr1 and addr1 == addr2 else 0.0
    ratio_n = 2.0 * common_n / (len(t1) + len(t2)) if (len(t1) + len(t2)) > 0 else 0.0
    ratio_a = 2.0 * common_a / (len(at1) + len(at2)) if (len(at1) + len(at2)) > 0 else 0.0

    return [
        lev_n,
        jacc_n,
        cos_n,
        lev_a,
        jacc_a,
        cos_a,
        float(common_n),
        float(common_a),
        float(abs(nl1 - nl2)),
        float(abs(al1 - al2)),
        exact_n,
        exact_a,
        ratio_n,
        ratio_a,
    ]


def extract_batch_features(
    s1_names: Sequence[str],
    s1_addrs: Sequence[str],
    cand_names: Sequence[str],
    cand_addrs: Sequence[str],
) -> np.ndarray:
    """Vectorized feature extraction for a batch of candidate pairs."""
    features = [
        extract_pair_features(n1, a1, n2, a2)
        for n1, a1, n2, a2 in zip(s1_names, s1_addrs, cand_names, cand_addrs)
    ]
    return np.array(features, dtype=np.float32)
