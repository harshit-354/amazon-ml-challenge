"""Evaluate rarity-weighted blocking on held-out labeled training queries."""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import pandas as pd

from src.normalization import normalize_dataframe
from src.rarity_blocking import generate_rarity_from_tsv


def _count_tsv_records(path: str) -> int:
    with open(path, "rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default="dataset/student_resource/dataset/train")
    parser.add_argument("--cache-dir", default="data/normalized_cache")
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument("--max-document-frequency", type=float, default=0.02)
    parser.add_argument("--name-threshold", type=float, default=0.18)
    parser.add_argument("--address-threshold", type=float, default=0.18)
    parser.add_argument("--top-k-per-field", type=int, default=50)
    parser.add_argument("--max-candidates", type=int, default=100)
    args = parser.parse_args()
    if not 0.0 < args.holdout_fraction < 1.0:
        parser.error("--holdout-fraction must be between 0 and 1")

    train_dir = args.train_dir.rstrip("/\\")
    source1_path = os.path.join(train_dir, "train_source1.tsv")
    source2_path = os.path.join(train_dir, "train_source2.tsv")
    source3_path = os.path.join(train_dir, "train_source3.tsv")
    ground_truth_path = os.path.join(train_dir, "train_ground_truth.tsv")
    print(f"[eval] Loading Source 1 IDs and ground truth from {train_dir}", flush=True)
    source1 = pd.read_csv(
        source1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id", "business_name", "business_address", "country"],
    )
    ground_truth_df = pd.read_csv(
        ground_truth_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    ground_truth = {
        str(row.source1_entity_id): {
            entity_id for entity_id in str(row.matched_entity_ids).split(",") if entity_id
        }
        for row in ground_truth_df.itertuples(index=False)
    }

    rng = np.random.default_rng(args.seed)
    holdout_size = max(1, int(round(len(source1) * args.holdout_fraction)))
    holdout_positions = np.sort(rng.choice(len(source1), size=holdout_size, replace=False))
    heldout = source1.iloc[holdout_positions].reset_index(drop=True)
    heldout = normalize_dataframe(heldout, verbose=True)
    print(
        f"[eval] Holdout: {len(heldout):,}/{len(source1):,} Source 1 queries "
        f"(seed={args.seed}); all S2/S3 targets remain searchable",
        flush=True,
    )

    os.makedirs(args.cache_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="blocking_eval_", dir=args.cache_dir) as temp_dir:
        heldout_path = os.path.join(temp_dir, "heldout_source1.tsv")
        candidate_path = os.path.join(temp_dir, "heldout_candidates.tsv")
        heldout.to_csv(heldout_path, sep="\t", index=False, encoding="utf-8")
        del heldout
        generate_rarity_from_tsv(
            heldout_path,
            source2_path,
            source3_path,
            candidate_path,
            cache_dir=args.cache_dir,
            chunksize=args.chunksize,
            max_document_frequency=args.max_document_frequency,
            name_threshold=args.name_threshold,
            address_threshold=args.address_threshold,
            top_k_per_field=args.top_k_per_field,
            max_candidates=args.max_candidates,
        )

        target_count = _count_tsv_records(source2_path) + _count_tsv_records(source3_path)
        possible_pairs = len(heldout) * target_count
        candidate_count = 0
        true_edges = 0
        retrieved_edges = 0
        positive_queries = 0
        hit_queries = 0
        reciprocal_rank_sum = 0.0
        singleton_queries = 0
        singleton_queries_with_candidates = 0

        for chunk in pd.read_csv(
            candidate_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=args.chunksize,
        ):
            for row in chunk.itertuples(index=False):
                source1_id = str(row.source1_entity_id)
                candidates = str(row.candidate_entity_ids).split(",") if row.candidate_entity_ids else []
                candidate_set = set(candidates)
                true_ids = ground_truth.get(source1_id, set())
                candidate_count += len(candidates)

                if not true_ids:
                    singleton_queries += 1
                    singleton_queries_with_candidates += bool(candidates)
                    continue

                positive_queries += 1
                true_edges += len(true_ids)
                found = true_ids.intersection(candidate_set)
                retrieved_edges += len(found)
                if found:
                    hit_queries += 1
                    first_correct_rank = next(
                        rank for rank, candidate_id in enumerate(candidates, start=1)
                        if candidate_id in true_ids
                    )
                    reciprocal_rank_sum += 1.0 / first_correct_rank

    edge_recall = retrieved_edges / true_edges if true_edges else 0.0
    query_recall = hit_queries / positive_queries if positive_queries else 0.0
    mrr = reciprocal_rank_sum / positive_queries if positive_queries else 0.0
    average_candidates = candidate_count / len(heldout) if len(heldout) else 0.0
    reduction_ratio = 1.0 - candidate_count / possible_pairs if possible_pairs else 0.0
    singleton_candidate_rate = (
        singleton_queries_with_candidates / singleton_queries if singleton_queries else 0.0
    )

    print("\nRarity-weighted held-out blocking evaluation")
    print(f"  Positive queries with a retrieved true match: {hit_queries:,}/{positive_queries:,} ({query_recall:.2%})")
    print(f"  Ground-truth match edges retrieved:           {retrieved_edges:,}/{true_edges:,} ({edge_recall:.2%})")
    print(f"  MRR (ranked candidate order):                 {mrr:.4f}")
    print(f"  Average candidates per query:                 {average_candidates:.2f}")
    print(f"  Candidate reduction ratio:                   {reduction_ratio:.2%}")
    print(
        f"  Singletons receiving any candidate:          "
        f"{singleton_queries_with_candidates:,}/{singleton_queries:,} ({singleton_candidate_rate:.2%})"
    )


if __name__ == "__main__":
    main()
