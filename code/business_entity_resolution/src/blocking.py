"""CLI and public API for rarity-weighted candidate generation."""

from __future__ import annotations

import argparse

import pandas as pd

from src.normalization import normalize_dataframe


def _prepare_frame(frame: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Ensure normalized fields exist for the in-memory DataFrame API."""
    required = {"entity_id", "business_name", "business_address"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source_label} is missing required columns: {sorted(missing)}")
    if {"norm_name", "norm_address"}.issubset(frame.columns):
        print(f"[blocking] {source_label}: normalized columns already present", flush=True)
        return frame
    print(f"[blocking] {source_label}: normalizing {len(frame):,} rows", flush=True)
    try:
        result = normalize_dataframe(frame, verbose=True)
    except Exception as exc:
        raise RuntimeError(f"Normalization failed for {source_label}: {exc}") from exc
    print(f"[blocking] {source_label}: normalization complete", flush=True)
    return result


def generate_candidate_pairs(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    *,
    max_document_frequency: float = 0.02,
    name_threshold: float = 0.18,
    address_threshold: float = 0.18,
    top_k_per_field: int = 50,
    max_candidates: int = 100,
) -> pd.DataFrame:
    """Generate rarity-weighted candidate lists from in-memory DataFrames.

    For the full challenge datasets, prefer :func:`generate_from_tsv`, which
    normalizes and indexes the target files in chunks with compact postings.
    """
    from src.rarity_blocking import generate_rarity_from_frames

    return generate_rarity_from_frames(
        source1,
        source2,
        source3,
        max_document_frequency=max_document_frequency,
        name_threshold=name_threshold,
        address_threshold=address_threshold,
        top_k_per_field=top_k_per_field,
        max_candidates=max_candidates,
    )


def generate_from_tsv(
    source1_path: str,
    source2_path: str,
    source3_path: str,
    output_path: str,
    *,
    cache_dir: str = "data/normalized_cache",
    chunksize: int = 200_000,
    max_document_frequency: float = 0.02,
    name_threshold: float = 0.18,
    address_threshold: float = 0.18,
    top_k_per_field: int = 50,
    max_candidates: int = 100,
) -> int:
    """Generate candidates from TSV paths without loading all records at once."""
    from src.rarity_blocking import generate_rarity_from_tsv

    return generate_rarity_from_tsv(
        source1_path,
        source2_path,
        source3_path,
        output_path,
        cache_dir=cache_dir,
        chunksize=chunksize,
        max_document_frequency=max_document_frequency,
        name_threshold=name_threshold,
        address_threshold=address_threshold,
        top_k_per_field=top_k_per_field,
        max_candidates=max_candidates,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate candidate pairs with a rarity-weighted token index"
    )
    parser.add_argument("--source1", required=True, help="Source 1 TSV path")
    parser.add_argument("--source2", required=True, help="Source 2 TSV path")
    parser.add_argument("--source3", required=True, help="Source 3 TSV path")
    parser.add_argument("--output", required=True, help="Output candidate_pairs.tsv path")
    parser.add_argument("--cache-dir", default="data/normalized_cache")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--max-document-frequency", type=float, default=0.02)
    parser.add_argument("--name-threshold", type=float, default=0.18)
    parser.add_argument("--address-threshold", type=float, default=0.18)
    parser.add_argument("--top-k-per-field", type=int, default=50)
    parser.add_argument("--max-candidates", type=int, default=100)
    args = parser.parse_args()

    generate_from_tsv(
        args.source1,
        args.source2,
        args.source3,
        args.output,
        cache_dir=args.cache_dir,
        chunksize=args.chunksize,
        max_document_frequency=args.max_document_frequency,
        name_threshold=args.name_threshold,
        address_threshold=args.address_threshold,
        top_k_per_field=args.top_k_per_field,
        max_candidates=args.max_candidates,
    )


if __name__ == "__main__":
    main()
