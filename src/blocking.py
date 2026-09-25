"""Candidate generation for the business entity resolution challenge.

Candidates are the union of three independent retrieval passes:

* token Jaccard overlap on normalized business names;
* token Jaccard overlap on normalized addresses;
* character n-gram Jaccard overlap on normalized business names.

Country is deliberately not consulted. Each pass uses an inverted index so
records with no shared tokens/n-grams are never compared pairwise.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Sequence

import pandas as pd

from src.normalization import normalize_dataframe


def _tokens(value: object) -> set[str]:
    if not isinstance(value, str) or not value:
        return set()
    return set(value.split())


def _char_ngrams(value: object, n: int = 3) -> set[str]:
    """Return padded character n-grams, retaining information at word edges."""
    if not isinstance(value, str) or not value:
        return set()
    padded = f"  {value}  "
    return {padded[i : i + n] for i in range(len(padded) - n + 1)}


def _prepare_frame(frame: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Ensure the normalized fields needed for blocking exist."""
    required = {"entity_id", "business_name", "business_address"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Source frame is missing required columns: {sorted(missing)}")

    if {"norm_name", "norm_address"}.issubset(frame.columns):
        print(f"[blocking] {source_label}: normalized columns already present; skipping normalization", flush=True)
        return frame
    print(f"[blocking] {source_label}: starting normalization ({len(frame):,} rows)", flush=True)
    try:
        result = normalize_dataframe(frame, verbose=True)
    except Exception as exc:
        raise RuntimeError(f"Normalization failed for {source_label}: {exc}") from exc
    print(f"[blocking] {source_label}: normalization complete", flush=True)
    return result


def _build_index(records: Sequence[set[str]]) -> dict[str, list[int]]:
    postings: dict[str, list[int]] = defaultdict(list)
    for row_number, values in enumerate(records):
        for value in values:
            postings[value].append(row_number)
    return postings


def _retrieve(
    query_values: set[str],
    records: Sequence[set[str]],
    postings: dict[str, list[int]],
    *,
    min_jaccard: float,
    min_shared: int,
    max_candidates: int,
) -> list[int]:
    """Retrieve indexed records meeting overlap criteria; rank by Jaccard."""
    if not query_values:
        return []
    shared: dict[int, int] = defaultdict(int)
    for value in query_values:
        for row_number in postings.get(value, ()):
            shared[row_number] += 1

    ranked: list[tuple[float, int, int]] = []
    for row_number, common in shared.items():
        candidate_values = records[row_number]
        union_size = len(query_values) + len(candidate_values) - common
        similarity = common / union_size if union_size else 0.0
        if common >= min_shared and similarity >= min_jaccard:
            ranked.append((similarity, common, row_number))

    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [row_number for _, _, row_number in ranked[:max_candidates]]


def generate_candidate_pairs(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    *,
    token_jaccard_threshold: float = 0.20,
    char_jaccard_threshold: float = 0.25,
    min_shared_tokens: int = 1,
    max_candidates_per_method: int = 100,
    char_ngram_size: int = 3,
) -> pd.DataFrame:
    """Generate candidate ID lists for every Source 1 entity.

    Thresholds apply separately to the token-name, token-address, and
    character-name passes. The returned pairs are their deduplicated union.
    ``max_candidates_per_method`` bounds unusually common tokens/grams while
    preserving the highest-Jaccard results from each pass.

    Returns a DataFrame with the submission-ready columns
    ``source1_entity_id`` and ``candidate_entity_ids``.
    """
    if not 0 <= token_jaccard_threshold <= 1 or not 0 <= char_jaccard_threshold <= 1:
        raise ValueError("Jaccard thresholds must be between 0 and 1")
    if min_shared_tokens < 1 or max_candidates_per_method < 1 or char_ngram_size < 1:
        raise ValueError("min_shared_tokens, max_candidates_per_method and char_ngram_size must be positive")

    s1 = _prepare_frame(source1, "Source 1").reset_index(drop=True)
    s2 = _prepare_frame(source2, "Source 2").reset_index(drop=True)
    s3 = _prepare_frame(source3, "Source 3").reset_index(drop=True)
    targets = pd.concat([s2, s3], ignore_index=True)

    print(f"[blocking] Building token and character indexes for {len(targets):,} target records", flush=True)
    name_tokens = [_tokens(value) for value in targets["norm_name"]]
    address_tokens = [_tokens(value) for value in targets["norm_address"]]
    name_grams = [_char_ngrams(value, char_ngram_size) for value in targets["norm_name"]]
    name_index = _build_index(name_tokens)
    address_index = _build_index(address_tokens)
    char_index = _build_index(name_grams)
    print("[blocking] Indexes ready; generating candidates", flush=True)

    target_ids = targets["entity_id"].astype(str).tolist()
    rows: list[dict[str, str]] = []
    total_source1 = len(s1)
    for row_number, (_, record) in enumerate(s1.iterrows(), start=1):
        candidate_rows: set[int] = set()
        query_name = _tokens(record["norm_name"])
        query_address = _tokens(record["norm_address"])

        candidate_rows.update(_retrieve(
            query_name, name_tokens, name_index,
            min_jaccard=token_jaccard_threshold,
            min_shared=min_shared_tokens,
            max_candidates=max_candidates_per_method,
        ))
        candidate_rows.update(_retrieve(
            query_address, address_tokens, address_index,
            min_jaccard=token_jaccard_threshold,
            min_shared=min_shared_tokens,
            max_candidates=max_candidates_per_method,
        ))
        candidate_rows.update(_retrieve(
            _char_ngrams(record["norm_name"], char_ngram_size),
            name_grams,
            char_index,
            min_jaccard=char_jaccard_threshold,
            min_shared=1,
            max_candidates=max_candidates_per_method,
        ))

        candidate_ids = sorted(target_ids[i] for i in candidate_rows)
        rows.append({
            "source1_entity_id": str(record["entity_id"]),
            "candidate_entity_ids": ",".join(candidate_ids),
        })
        if row_number % 10000 == 0 or row_number == total_source1:
            print(f"[blocking] Candidate generation: {row_number:,}/{total_source1:,} Source 1 records", flush=True)

    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])


def generate_from_tsv(
    source1_path: str,
    source2_path: str,
    source3_path: str,
    output_path: str,
    **kwargs: object,
) -> pd.DataFrame:
    """Read three challenge TSVs, generate candidates and write the TSV output."""
    print(f"[blocking] Reading Source 1: {source1_path}", flush=True)
    source1 = pd.read_csv(source1_path, sep="\t", dtype=str, keep_default_na=False)
    print(f"[blocking] Read {len(source1):,} Source 1 records", flush=True)
    print(f"[blocking] Reading Source 2: {source2_path}", flush=True)
    source2 = pd.read_csv(source2_path, sep="\t", dtype=str, keep_default_na=False)
    print(f"[blocking] Read {len(source2):,} Source 2 records", flush=True)
    print(f"[blocking] Reading Source 3: {source3_path}", flush=True)
    source3 = pd.read_csv(source3_path, sep="\t", dtype=str, keep_default_na=False)
    print(f"[blocking] Read {len(source3):,} Source 3 records", flush=True)
    candidates = generate_candidate_pairs(source1, source2, source3, **kwargs)
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    print(f"[blocking] Writing {len(candidates):,} rows to {output_path}", flush=True)
    candidates.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    print("[blocking] Output write complete", flush=True)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cross-source entity-resolution candidates")
    parser.add_argument("--source1", required=True, help="Source 1 TSV path")
    parser.add_argument("--source2", required=True, help="Source 2 TSV path")
    parser.add_argument("--source3", required=True, help="Source 3 TSV path")
    parser.add_argument("--output", required=True, help="Output candidate_pairs.tsv path")
    parser.add_argument("--token-threshold", type=float, default=0.20)
    parser.add_argument("--char-threshold", type=float, default=0.25)
    parser.add_argument("--max-per-method", type=int, default=100)
    args = parser.parse_args()
    result = generate_from_tsv(
        args.source1, args.source2, args.source3, args.output,
        token_jaccard_threshold=args.token_threshold,
        char_jaccard_threshold=args.char_threshold,
        max_candidates_per_method=args.max_per_method,
    )
    print(f"Wrote candidates for {len(result):,} Source 1 entities to {args.output}")


if __name__ == "__main__":
    main()
