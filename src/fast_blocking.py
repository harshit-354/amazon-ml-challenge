"""High-throughput multi-core candidate generation with resume support."""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import os
import sys
import time
from typing import Iterator

import pandas as pd

from src.inference import run_inference
from src.rarity_blocking import (
    _RarityIndex,
    _frames_from_tsv,
    ensure_normalized_tsv,
)

# Global index instance inherited by forked workers
_WORKER_INDEX: _RarityIndex | None = None
_WORKER_PARAMS: dict[str, float | int] = {}


def _worker_init(index: _RarityIndex, params: dict[str, float | int]) -> None:
    global _WORKER_INDEX, _WORKER_PARAMS
    _WORKER_INDEX = index
    _WORKER_PARAMS = params


def _process_query_batch(batch: list[tuple[str, str, str]]) -> list[tuple[str, str]]:
    global _WORKER_INDEX, _WORKER_PARAMS
    assert _WORKER_INDEX is not None
    n_thresh = float(_WORKER_PARAMS["name_threshold"])
    a_thresh = float(_WORKER_PARAMS["address_threshold"])
    top_k = int(_WORKER_PARAMS["top_k_per_field"])
    max_cands = int(_WORKER_PARAMS["max_candidates"])

    results: list[tuple[str, str]] = []
    for entity_id, norm_name, norm_addr in batch:
        scored = _WORKER_INDEX.candidates(
            norm_name,
            norm_addr,
            name_threshold=n_thresh,
            address_threshold=a_thresh,
            top_k_per_field=top_k,
            max_candidates=max_cands,
        )
        results.append((str(entity_id), ",".join(scored)))
    return results


def _count_existing_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        # Count lines excluding header
        count = sum(1 for _ in f) - 1
        return max(0, count)


def _batch_generator(
    source1_norm: str,
    skip_rows: int,
    batch_size: int = 1_000,
) -> Iterator[list[tuple[str, str, str]]]:
    """Stream Source 1 records in batches of tuples, skipping already processed rows."""
    with open(source1_norm, "r", encoding="utf-8") as f:
        header = next(f).rstrip("\r\n").split("\t")
        col_to_idx = {col: i for i, col in enumerate(header)}
        eid_idx = col_to_idx["entity_id"]
        name_idx = col_to_idx["norm_name"]
        addr_idx = col_to_idx["norm_address"]

        # Fast skip
        for _ in range(skip_rows):
            next(f, None)

        batch: list[tuple[str, str, str]] = []
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            eid = parts[eid_idx]
            name = parts[name_idx] if name_idx < len(parts) else ""
            addr = parts[addr_idx] if addr_idx < len(parts) else ""
            batch.append((eid, name, addr))

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


def run_fast_blocking(
    source1_path: str = "student_resource/dataset/test/test_source1.tsv",
    source2_path: str = "student_resource/dataset/test/test_source2.tsv",
    source3_path: str = "student_resource/dataset/test/test_source3.tsv",
    candidate_output: str = "output/candidate_pairs.tsv",
    cache_dir: str = "data/normalized_cache",
    num_workers: int = 5,
    max_document_frequency: float = 0.02,
    name_threshold: float = 0.18,
    address_threshold: float = 0.18,
    top_k_per_field: int = 50,
    max_candidates: int = 100,
    batch_size: int = 1_000,
) -> None:
    t_start = time.time()
    cache_dir = os.path.abspath(cache_dir)
    print(f"[fast_blocking] Ensuring normalized caches in {cache_dir}...", flush=True)
    source1_norm = ensure_normalized_tsv(source1_path, cache_dir=cache_dir)
    source2_norm = ensure_normalized_tsv(source2_path, cache_dir=cache_dir)
    source3_norm = ensure_normalized_tsv(source3_path, cache_dir=cache_dir)

    # 1. Build inverted index over S2 & S3 targets
    target_factory = lambda: itertools.chain(
        _frames_from_tsv(source2_norm, chunksize=200_000),
        _frames_from_tsv(source3_norm, chunksize=200_000),
    )
    print("[fast_blocking] Building rarity inverted index...", flush=True)
    index = _RarityIndex(
        target_factory,
        max_document_frequency=max_document_frequency,
        verbose=True,
    )

    # 2. Check resume status
    existing_rows = _count_existing_rows(candidate_output)
    total_s1 = sum(1 for _ in open(source1_norm, "rb")) - 1

    mode = "a" if existing_rows > 0 else "w"
    print(
        f"\n[fast_blocking] Status: {existing_rows:,}/{total_s1:,} queries already generated in {candidate_output}.\n"
        f"  Resuming remaining {total_s1 - existing_rows:,} queries across {num_workers} parallel workers...\n",
        flush=True,
    )

    params = {
        "name_threshold": name_threshold,
        "address_threshold": address_threshold,
        "top_k_per_field": top_k_per_field,
        "max_candidates": max_candidates,
    }

    # 3. Setup multiprocessing fork pool
    ctx = mp.get_context("fork")
    processed_count = existing_rows
    t_resume = time.time()

    with ctx.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(index, params),
    ) as pool, open(candidate_output, mode, encoding="utf-8", newline="") as out_f:

        writer = csv.writer(out_f, delimiter="\t", lineterminator="\n")
        if existing_rows == 0:
            writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        batch_iter = _batch_generator(source1_norm, skip_rows=existing_rows, batch_size=batch_size)

        for batch_results in pool.imap(_process_query_batch, batch_iter, chunksize=1):
            for eid, cands_str in batch_results:
                writer.writerow([eid, cands_str])
            processed_count += len(batch_results)

            if processed_count % 25_000 == 0 or processed_count == total_s1:
                elapsed = time.time() - t_resume
                rate = (processed_count - existing_rows) / max(elapsed, 0.001)
                remaining_time = (total_s1 - processed_count) / max(rate, 0.001)
                print(
                    f"[fast_blocking] Progress: {processed_count:,}/{total_s1:,} queries "
                    f"({rate:.0f} q/s | {elapsed/60:.1f}m elapsed | ~{remaining_time/60:.1f}m remaining)",
                    flush=True,
                )

    print(
        f"\n[fast_blocking] Candidate pairs complete! Total rows in {candidate_output}: {processed_count:,}. "
        f"Elapsed time: {time.time()-t_start:.1f}s\n",
        flush=True,
    )

    # 4. Immediately trigger inference
    print("[fast_blocking] Launching model inference to generate output/matching_results.tsv...", flush=True)
    run_inference(
        candidate_pairs_path=candidate_output,
        test_dir=os.path.dirname(os.path.abspath(source1_path)),
        cache_dir=cache_dir,
        model_path="models/matching_model.pkl",
        output_path="output/matching_results.tsv",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-core fast candidate generation and inference.")
    parser.add_argument("--source1", default="student_resource/dataset/test/test_source1.tsv")
    parser.add_argument("--source2", default="student_resource/dataset/test/test_source2.tsv")
    parser.add_argument("--source3", default="student_resource/dataset/test/test_source3.tsv")
    parser.add_argument("--output", default="output/candidate_pairs.tsv")
    parser.add_argument("--cache-dir", default="data/normalized_cache")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    run_fast_blocking(
        source1_path=args.source1,
        source2_path=args.source2,
        source3_path=args.source3,
        candidate_output=args.output,
        cache_dir=args.cache_dir,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
