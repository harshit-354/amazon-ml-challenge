"""High-throughput multi-core inference pipeline for generating high-precision matching results."""

from __future__ import annotations

import argparse
import csv
import glob
import multiprocessing as mp
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import jellyfish
import numpy as np

from src.features import extract_pair_features
from src.normalization import normalize_name, normalize_address

# Shared global structures inherited by forked workers
_GLOBAL_S1: dict[str, tuple[str, str, str]] = {}
_GLOBAL_TARGETS: dict[str, tuple[str, str, str]] = {}
_GLOBAL_MODEL: Any = None
_GLOBAL_THRESHOLD: float = 0.82
_GLOBAL_MAX_MATCHES: int = 6


def _find_normalized_cache(prefix: str, cache_dir: str = "data/normalized_cache") -> str | None:
    matches = glob.glob(os.path.join(cache_dir, f"{prefix}.*.normalized.tsv"))
    return matches[0] if matches else None


def load_source_records(
    source_path: str,
    prefix: str,
    cache_dir: str = "data/normalized_cache",
) -> dict[str, tuple[str, str, str]]:
    """Load normalized (norm_name, norm_address, country) for entity records."""
    cache_file = _find_normalized_cache(prefix, cache_dir)
    use_path = cache_file if cache_file and os.path.exists(cache_file) else source_path
    print(f"[inference] Loading records for {prefix} from {use_path}...", flush=True)

    records: dict[str, tuple[str, str, str]] = {}
    is_cached = use_path.endswith(".normalized.tsv")

    with open(use_path, "r", encoding="utf-8") as f:
        header_line = next(f)
        header = header_line.rstrip("\r\n").split("\t")
        col_to_idx = {col: i for i, col in enumerate(header)}

        eid_idx = col_to_idx.get("entity_id", 0)
        name_idx = col_to_idx.get("norm_name" if is_cached else "business_name", 1)
        addr_idx = col_to_idx.get("norm_address" if is_cached else "business_address", 2)
        country_idx = col_to_idx.get("country", 3 if len(header) > 3 else -1)

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            eid = parts[eid_idx]
            name = parts[name_idx] if name_idx < len(parts) else ""
            addr = parts[addr_idx] if addr_idx < len(parts) else ""
            country = parts[country_idx] if (0 <= country_idx < len(parts)) else ""

            if not is_cached:
                name = normalize_name(name)
                addr = normalize_address(addr)

            records[eid] = (name, addr, country)

    print(f"[inference] Loaded {len(records):,} records for {prefix}", flush=True)
    return records


def _worker_process_chunk(
    worker_id: int,
    candidate_pairs_path: str,
    start_line: int,
    end_line: int,
    part_output_path: str,
    batch_size: int = 50_000,
) -> tuple[int, int, int, int, int]:
    """Worker function executed by each forked child process."""
    re_digits = re.compile(r"\b\d+\b")

    total_s1 = 0
    total_singletons = 0
    total_matches = 0
    pairs_scored = 0

    batch_pairs: list[tuple[str, str, list[float]]] = []
    batch_s1_order: list[tuple[str, list[str]]] = []

    with open(candidate_pairs_path, "r", encoding="utf-8") as in_f, \
         open(part_output_path, "w", encoding="utf-8", newline="") as out_f:

        writer = csv.writer(out_f, delimiter="\t", lineterminator="\n")

        # Skip header and fast-forward to start_line (1-indexed data lines)
        next(in_f)  # header
        for _ in range(start_line):
            next(in_f, None)

        def flush_batch() -> None:
            nonlocal total_matches, total_singletons, total_s1, pairs_scored
            if not batch_s1_order:
                return

            s1_to_matched: dict[str, list[tuple[str, float]]] = {s1: [] for s1, _ in batch_s1_order}
            if batch_pairs:
                feats = np.array([p[2] for p in batch_pairs], dtype=np.float32)
                probs = _GLOBAL_MODEL.predict_proba(feats)[:, 1]
                pairs_scored += len(probs)
                for (s1_id, cand_id, _), prob in zip(batch_pairs, probs):
                    if prob >= _GLOBAL_THRESHOLD:
                        s1_to_matched[s1_id].append((cand_id, float(prob)))

            for s1_id, _ in batch_s1_order:
                matched_items = s1_to_matched.get(s1_id, [])
                if matched_items:
                    matched_items.sort(key=lambda x: -x[1])
                    seen: set[str] = set()
                    final_ids: list[str] = []
                    for cid, _ in matched_items:
                        if cid not in seen:
                            seen.add(cid)
                            final_ids.append(cid)
                        if len(final_ids) >= _GLOBAL_MAX_MATCHES:
                            break
                    writer.writerow([s1_id, ",".join(final_ids)])
                    total_matches += len(final_ids)
                else:
                    writer.writerow([s1_id, ""])
                    total_singletons += 1
                total_s1 += 1

            batch_pairs.clear()
            batch_s1_order.clear()

        lines_to_read = end_line - start_line
        for line_no, line in enumerate(in_f):
            if line_no >= lines_to_read:
                break

            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            cand_str = parts[1] if len(parts) > 1 else ""
            candidates = [c.strip() for c in cand_str.split(",") if c.strip()] if cand_str else []

            batch_s1_order.append((s1_id, candidates))

            if candidates:
                s1_name, s1_addr, s1_country = _GLOBAL_S1.get(s1_id, ("", "", ""))
                s1_nums = set(re_digits.findall(s1_addr))
                s1_ntoks = set(s1_name.split())
                s1_atoks = set(s1_addr.split())

                for cand_id in candidates:
                    if cand_id not in _GLOBAL_TARGETS:
                        continue
                    cand_name, cand_addr, cand_country = _GLOBAL_TARGETS[cand_id]

                    # 1. Country constraint: strict match
                    if s1_country and cand_country and s1_country != cand_country:
                        continue

                    cand_ntoks = set(cand_name.split())
                    cand_atoks = set(cand_addr.split())
                    common_n = len(s1_ntoks & cand_ntoks)
                    common_a = len(s1_atoks & cand_atoks)

                    if not common_n and not common_a:
                        continue

                    # 2. Number conflict check
                    cand_nums = set(re_digits.findall(cand_addr))
                    numbers_match = bool(s1_nums and cand_nums and (s1_nums & cand_nums))
                    number_conflict = bool(s1_nums and cand_nums and not (s1_nums & cand_nums))

                    # 3. Extract similarity features
                    feat = extract_pair_features(s1_name, s1_addr, cand_name, cand_addr)
                    name_max = max(feat[0], feat[1], feat[12])
                    lev_a = feat[3]

                    # 4. Precision gating rules
                    if number_conflict and name_max < 0.60:
                        continue
                    if not cand_addr and name_max < 0.60:
                        continue
                    if common_n == 0 and name_max < 0.40:
                        if not (numbers_match and lev_a >= 0.85):
                            continue
                    if name_max < 0.35 and lev_a < 0.85:
                        continue

                    batch_pairs.append((s1_id, cand_id, feat))

            if len(batch_pairs) >= batch_size:
                flush_batch()

        flush_batch()

    print(
        f"[worker {worker_id}] Finished {total_s1:,} queries "
        f"({total_matches:,} matches, {total_singletons:,} singletons, {pairs_scored:,} scored)",
        flush=True,
    )
    return worker_id, total_s1, total_singletons, total_matches, pairs_scored


def run_inference(
    candidate_pairs_path: str,
    test_dir: str = "student_resource/dataset/test",
    cache_dir: str = "data/normalized_cache",
    model_path: str = "models/matching_model.pkl",
    output_path: str = "output/matching_results.tsv",
    batch_size: int = 50_000,
    max_matches_per_entity: int = 6,
    num_workers: int = 6,
) -> dict[str, Any]:
    """Score candidate pairs using multi-process fork parallelism."""
    global _GLOBAL_S1, _GLOBAL_TARGETS, _GLOBAL_MODEL, _GLOBAL_THRESHOLD, _GLOBAL_MAX_MATCHES

    t_start = time.time()
    print(f"[inference] Loading trained model artifact from {model_path}...", flush=True)
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    _GLOBAL_MODEL = artifact["model"]
    _GLOBAL_THRESHOLD = float(artifact.get("optimal_threshold", 0.82))
    _GLOBAL_MAX_MATCHES = int(max_matches_per_entity)
    print(
        f"[inference] Model loaded. Decision threshold: {_GLOBAL_THRESHOLD:.2f} | "
        f"Max matches: {_GLOBAL_MAX_MATCHES}",
        flush=True,
    )

    # 1. Load S1 reference entities
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    _GLOBAL_S1 = load_source_records(s1_path, "test_source1", cache_dir)

    # 2. Load target entities (S2 & S3)
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    s2_records = load_source_records(s2_path, "test_source2", cache_dir)
    s3_records = load_source_records(s3_path, "test_source3", cache_dir)

    _GLOBAL_TARGETS = {}
    _GLOBAL_TARGETS.update(s2_records)
    _GLOBAL_TARGETS.update(s3_records)
    del s2_records, s3_records

    print(
        f"[inference] Target entities index ready: {len(_GLOBAL_TARGETS):,} records. "
        f"Index setup took {time.time()-t_start:.1f}s.",
        flush=True,
    )

    # 3. Count total query lines in candidate_pairs.tsv
    print(f"[inference] Counting queries in {candidate_pairs_path}...", flush=True)
    total_queries = 0
    with open(candidate_pairs_path, "r", encoding="utf-8") as f:
        next(f)  # header
        for _ in f:
            total_queries += 1

    print(
        f"[inference] Total Source 1 queries to process: {total_queries:,}. "
        f"Spawning {num_workers} parallel workers...",
        flush=True,
    )

    chunk_size = (total_queries + num_workers - 1) // num_workers
    part_files: list[str] = []
    worker_args = []

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    for w_id in range(num_workers):
        start = w_id * chunk_size
        end = min((w_id + 1) * chunk_size, total_queries)
        if start >= total_queries:
            break
        part_path = os.path.join(output_dir, f"_matching_part_{w_id}.tsv")
        part_files.append(part_path)
        worker_args.append((
            w_id,
            candidate_pairs_path,
            start,
            end,
            part_path,
            batch_size,
        ))

    # Fork workers sharing memory via Copy-on-Write
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=len(worker_args)) as pool:
        results = pool.starmap(_worker_process_chunk, worker_args)

    total_s1 = sum(r[1] for r in results)
    total_singletons = sum(r[2] for r in results)
    total_matches = sum(r[3] for r in results)
    total_pairs_scored = sum(r[4] for r in results)

    # Concatenate part files into final output
    print(f"[inference] Merging {len(part_files)} worker parts into {output_path}...", flush=True)
    with open(output_path, "w", encoding="utf-8", newline="") as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        for part_path in part_files:
            with open(part_path, "r", encoding="utf-8") as p_f:
                shutil.copyfileobj(p_f, out_f)
            os.remove(part_path)

    total_time = time.time() - t_start
    print(
        f"\n[inference] Successfully generated {output_path}!\n"
        f"  Total Source 1 Records:    {total_s1:,}\n"
        f"  Total Pairs Scored:        {total_pairs_scored:,}\n"
        f"  Total Matched Links:       {total_matches:,}\n"
        f"  Total Singletons (empty):  {total_singletons:,} ({total_singletons/max(total_s1,1):.2%})\n"
        f"  Average Matches/Entity:    {total_matches/max(total_s1,1):.2f}\n"
        f"  Total Inference Time:      {total_time:.1f}s ({total_s1/max(total_time,0.001):.0f} queries/sec)\n",
        flush=True,
    )

    # Validate output with official validator
    validator_path = "student_resource/utils/validate_submission.py"
    if os.path.exists(validator_path):
        print(f"[inference] Running submission validator ({validator_path})...", flush=True)
        cmd = [
            sys.executable,
            validator_path,
            "--matching", output_path,
            "--candidate", candidate_pairs_path,
            "--test-dir", test_dir,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr, file=sys.stderr)

    return {
        "output_path": output_path,
        "total_s1": total_s1,
        "total_matches": total_matches,
        "total_singletons": total_singletons,
        "total_time": total_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-core inference to generate matching_results.tsv.")
    parser.add_argument("--candidate-pairs", default="output/candidate_pairs.tsv")
    parser.add_argument("--test-dir", default="student_resource/dataset/test")
    parser.add_argument("--cache-dir", default="data/normalized_cache")
    parser.add_argument("--model", default="models/matching_model.pkl")
    parser.add_argument("--output", default="output/matching_results.tsv")
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--max-matches", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=6)
    args = parser.parse_args()

    run_inference(
        candidate_pairs_path=args.candidate_pairs,
        test_dir=args.test_dir,
        cache_dir=args.cache_dir,
        model_path=args.model,
        output_path=args.output,
        batch_size=args.batch_size,
        max_matches_per_entity=args.max_matches,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
