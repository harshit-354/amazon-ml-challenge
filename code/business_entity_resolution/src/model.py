"""Training and validation pipeline for business entity matching model."""

from __future__ import annotations

import argparse
import os
import pickle
import random
import time
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features import FEATURE_NAMES, extract_pair_features
from src.normalization import normalize_name, normalize_address


def compute_entity_f05(true_set: set[str], pred_set: set[str]) -> float:
    """Compute F_0.5 score for a single Source 1 entity."""
    if not true_set:
        # Singleton: 1.0 if correctly predicted empty, 0.0 if any false merge
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0
    tp = len(true_set.intersection(pred_set))
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    denom = 0.25 * precision + recall
    if denom <= 0:
        return 0.0
    return (1.25 * precision * recall) / denom


def evaluate_macro_f05(
    ground_truth: dict[str, set[str]],
    predictions: dict[str, set[str]],
) -> dict[str, float]:
    """Calculate macro-averaged F_0.5 across all entities in ground truth."""
    scores: list[float] = []
    singleton_total = 0
    singleton_correct = 0
    non_singleton_total = 0
    total_tp = 0
    total_pred = 0
    total_true = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = compute_entity_f05(true_set, pred_set)
        scores.append(score)

        if not true_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
        else:
            non_singleton_total += 1
            tp = len(true_set.intersection(pred_set))
            total_tp += tp
            total_pred += len(pred_set)
            total_true += len(true_set)

    macro_f05 = float(np.mean(scores)) if scores else 0.0
    prec = total_tp / total_pred if total_pred > 0 else 0.0
    rec = total_tp / total_true if total_true > 0 else 0.0
    singleton_acc = singleton_correct / singleton_total if singleton_total > 0 else 1.0

    return {
        "macro_f05": macro_f05,
        "precision": prec,
        "recall": rec,
        "singleton_accuracy": singleton_acc,
        "singleton_count": float(singleton_total),
        "non_singleton_count": float(non_singleton_total),
        "total_queries": float(len(scores)),
    }


def prepare_training_pairs(
    train_dir: str,
    num_queries: int = 25_000,
    seed: int = 42,
) -> tuple[
    dict[str, tuple[str, str]],
    dict[str, set[str]],
    dict[str, tuple[str, str]],
    list[tuple[str, str, str]],
]:
    """Load S1 queries, ground truth, and target records from S2/S3."""
    random.seed(seed)
    np.random.seed(seed)

    s1_path = os.path.join(train_dir, "train_source1.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")

    print(f"[model] 1/3: Reading up to {num_queries:,} Source 1 records...", flush=True)
    s1_records: dict[str, tuple[str, str]] = {}
    with open(s1_path, "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            eid = parts[0]
            name = normalize_name(parts[1]) if len(parts) > 1 else ""
            addr = normalize_address(parts[2]) if len(parts) > 2 else ""
            s1_records[eid] = (name, addr)
            if i + 1 >= num_queries:
                break

    print(f"[model] 2/3: Reading ground truth for sampled Source 1 records...", flush=True)
    gt: dict[str, set[str]] = {}
    needed_targets: set[str] = set()
    with open(gt_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            if s1_id in s1_records:
                matches = {m.strip() for m in parts[1].split(",") if m.strip()} if len(parts) > 1 and parts[1] else set()
                gt[s1_id] = matches
                needed_targets.update(matches)
            if len(gt) >= len(s1_records):
                break

    print(f"[model] 3/3: Scanning S2 and S3 for {len(needed_targets):,} positive targets + hard negatives...", flush=True)
    target_records: dict[str, tuple[str, str]] = {}
    neg_pool: list[tuple[str, str, str]] = []
    name_token_to_targets: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    addr_token_to_targets: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    for path in [s2_path, s3_path]:
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for i, line in enumerate(f):
                parts = line.strip().split("\t")
                eid = parts[0]
                is_target = eid in needed_targets
                sample_neg = len(neg_pool) < 25_000 and (i % 250 == 0)

                if is_target or sample_neg:
                    n = normalize_name(parts[1]) if len(parts) > 1 else ""
                    a = normalize_address(parts[2]) if len(parts) > 2 else ""

                    if is_target:
                        target_records[eid] = (n, a)

                    if sample_neg:
                        rec = (eid, n, a)
                        neg_pool.append(rec)
                        for t in n.split():
                            if len(t) > 3 and len(name_token_to_targets[t]) < 25:
                                name_token_to_targets[t].append(rec)
                        for t in a.split():
                            if len(t) > 3 and len(addr_token_to_targets[t]) < 25:
                                addr_token_to_targets[t].append(rec)

    print(
        f"[model] Target scan complete: {len(target_records):,} positive targets, "
        f"{len(neg_pool):,} negative pool candidates.",
        flush=True,
    )
    return s1_records, gt, target_records, neg_pool, name_token_to_targets, addr_token_to_targets


def build_labeled_dataset(
    s1_records: dict[str, tuple[str, str]],
    gt: dict[str, set[str]],
    target_records: dict[str, tuple[str, str]],
    neg_pool: list[tuple[str, str, str]],
    name_token_to_targets: dict[str, list[tuple[str, str, str]]],
    addr_token_to_targets: dict[str, list[tuple[str, str, str]]],
    val_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[
    np.ndarray, np.ndarray,
    dict[str, list[tuple[str, list[float]]]],
    dict[str, set[str]],
]:
    """Construct labeled feature arrays with hard negatives for training and validation."""
    random.seed(seed)
    s1_keys = list(s1_records.keys())
    random.shuffle(s1_keys)

    val_size = int(len(s1_keys) * val_fraction)
    val_keys = set(s1_keys[:val_size])
    train_keys = set(s1_keys[val_size:])

    X_train: list[list[float]] = []
    y_train: list[int] = []

    # Validation pairs: s1_id -> list of (target_id, features)
    val_candidates: dict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    val_gt: dict[str, set[str]] = {k: gt.get(k, set()) for k in val_keys}

    def sample_hard_negative(n1: str, a1: str, true_ids: set[str]) -> tuple[str, str, str]:
        # Try finding a name hard negative
        for t in n1.split():
            candidates = name_token_to_targets.get(t, [])
            for cand in candidates:
                if cand[0] not in true_ids:
                    return cand
        # Try finding an address hard negative
        for t in a1.split():
            candidates = addr_token_to_targets.get(t, [])
            for cand in candidates:
                if cand[0] not in true_ids:
                    return cand
        # Fallback to random negative
        return random.choice(neg_pool)

    # Populate training pairs
    for s1_id in train_keys:
        n1, a1 = s1_records[s1_id]
        true_ids = gt.get(s1_id, set())

        # Positives
        for tid in true_ids:
            if tid in target_records:
                n2, a2 = target_records[tid]
                X_train.append(extract_pair_features(n1, a1, n2, a2))
                y_train.append(1)

                # Hard negative (shares name/address token but different entity)
                hn_id, hn2, ha2 = sample_hard_negative(n1, a1, true_ids)
                X_train.append(extract_pair_features(n1, a1, hn2, ha2))
                y_train.append(0)

                # Random negative
                rn_id, rn2, ra2 = random.choice(neg_pool)
                X_train.append(extract_pair_features(n1, a1, rn2, ra2))
                y_train.append(0)

        # Singletons also get negative pairs
        if not true_ids:
            for _ in range(3):
                hn_id, hn2, ha2 = sample_hard_negative(n1, a1, true_ids)
                X_train.append(extract_pair_features(n1, a1, hn2, ha2))
                y_train.append(0)

    # Populate validation pairs (positives + hard negatives + random distractors)
    for s1_id in val_keys:
        n1, a1 = s1_records[s1_id]
        true_ids = gt.get(s1_id, set())
        for tid in true_ids:
            if tid in target_records:
                n2, a2 = target_records[tid]
                feat = extract_pair_features(n1, a1, n2, a2)
                val_candidates[s1_id].append((tid, feat))

        # Add 3 hard negative distractors
        for _ in range(3):
            hn_id, hn2, ha2 = sample_hard_negative(n1, a1, true_ids)
            feat = extract_pair_features(n1, a1, hn2, ha2)
            val_candidates[s1_id].append((hn_id, feat))

        # Add 2 random distractors
        for _ in range(2):
            rn_id, rn2, ra2 = random.choice(neg_pool)
            if rn_id not in true_ids:
                feat = extract_pair_features(n1, a1, rn2, ra2)
                val_candidates[s1_id].append((rn_id, feat))

    return (
        np.array(X_train, dtype=np.float32),
        np.array(y_train, dtype=np.int32),
        val_candidates,
        val_gt,
    )


def optimize_threshold(
    model: Any,
    val_candidates: dict[str, list[tuple[str, list[float]]]],
    val_gt: dict[str, set[str]],
) -> tuple[float, float, dict[float, float]]:
    """Sweep decision thresholds on validation pairs to maximize macro F_0.5."""
    flat_feats: list[list[float]] = []
    flat_meta: list[tuple[str, str]] = []
    for s1_id, pairs in val_candidates.items():
        for target_id, feat in pairs:
            flat_feats.append(feat)
            flat_meta.append((s1_id, target_id))

    all_s1_pairs: dict[str, list[tuple[str, float]]] = defaultdict(list)
    if flat_feats:
        flat_probs = model.predict_proba(np.array(flat_feats, dtype=np.float32))[:, 1]
        for (s1_id, target_id), prob in zip(flat_meta, flat_probs):
            all_s1_pairs[s1_id].append((target_id, float(prob)))

    best_thresh = 0.50
    best_f05 = -1.0
    threshold_scores: dict[float, float] = {}

    for thresh in np.arange(0.10, 0.95, 0.02):
        thresh = round(float(thresh), 2)
        preds: dict[str, set[str]] = {}
        for s1_id, true_set in val_gt.items():
            pairs = all_s1_pairs.get(s1_id, [])
            matched = {tid for tid, p in pairs if p >= thresh}
            preds[s1_id] = matched

        metrics = evaluate_macro_f05(val_gt, preds)
        score = metrics["macro_f05"]
        threshold_scores[thresh] = score
        if score > best_f05:
            best_f05 = score
            best_thresh = thresh

    return best_thresh, best_f05, threshold_scores, all_s1_pairs


def train_and_evaluate(
    train_dir: str = "student_resource/dataset/train",
    num_queries: int = 25_000,
    val_fraction: float = 0.20,
    output_model_path: str = "models/matching_model.pkl",
    seed: int = 42,
) -> tuple[HistGradientBoostingClassifier, float, dict[str, Any]]:
    """End-to-end training, threshold tuning, and model export."""
    t_start = time.time()
    (
        s1_records,
        gt,
        target_records,
        neg_pool,
        name_token_to_targets,
        addr_token_to_targets,
    ) = prepare_training_pairs(
        train_dir=train_dir,
        num_queries=num_queries,
        seed=seed,
    )

    X_train, y_train, val_candidates, val_gt = build_labeled_dataset(
        s1_records,
        gt,
        target_records,
        neg_pool,
        name_token_to_targets,
        addr_token_to_targets,
        val_fraction=val_fraction,
        seed=seed,
    )
    print(
        f"[model] Built training set: {len(X_train):,} pairs "
        f"({sum(y_train):,} positive, {len(y_train)-sum(y_train):,} negative)",
        flush=True,
    )

    print("[model] Training HistGradientBoostingClassifier...", flush=True)
    clf = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    print("[model] Optimizing F_0.5 decision threshold on validation queries...", flush=True)
    best_thresh, best_f05, sweep_results, all_s1_pairs = optimize_threshold(clf, val_candidates, val_gt)

    # Evaluate final validation metrics at best threshold
    all_preds: dict[str, set[str]] = {}
    for s1_id in val_gt:
        pairs = all_s1_pairs.get(s1_id, [])
        all_preds[s1_id] = {tid for tid, p in pairs if p >= best_thresh}

    final_metrics = evaluate_macro_f05(val_gt, all_preds)
    print(
        f"\n[model] Validation Results at optimal threshold ({best_thresh:.2f}):\n"
        f"  Macro F_0.5:         {final_metrics['macro_f05']:.4f}\n"
        f"  Precision:           {final_metrics['precision']:.4f}\n"
        f"  Recall:              {final_metrics['recall']:.4f}\n"
        f"  Singleton Accuracy:  {final_metrics['singleton_accuracy']:.2%}\n"
        f"  Total Val Queries:   {int(final_metrics['total_queries']):,}\n"
        f"  Total Time:          {time.time()-t_start:.1f}s\n",
        flush=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)
    model_artifact = {
        "model": clf,
        "optimal_threshold": best_thresh,
        "feature_names": FEATURE_NAMES,
        "val_metrics": final_metrics,
        "threshold_sweep": sweep_results,
    }
    with open(output_model_path, "wb") as f:
        pickle.dump(model_artifact, f)
    print(f"[model] Model artifact saved to {output_model_path}", flush=True)

    return clf, best_thresh, model_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and validate business entity matching model.")
    parser.add_argument("--train-dir", default="student_resource/dataset/train")
    parser.add_argument("--num-queries", type=int, default=25_000)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--output-model", default="models/matching_model.pkl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_and_evaluate(
        train_dir=args.train_dir,
        num_queries=args.num_queries,
        val_fraction=args.val_fraction,
        output_model_path=args.output_model,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
