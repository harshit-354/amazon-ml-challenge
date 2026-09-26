"""Chunked, rarity-weighted inverted-index candidate generation.

The index is built only from the supplied Source 2/3 records. It stores target
row numbers in compact unsigned-integer arrays and streams normalized TSVs,
avoiding the dense vectors and large per-token Python integer lists used by
the earlier blocking options.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import tempfile
from array import array
from collections import Counter, defaultdict
from itertools import chain
from typing import Callable, Iterable, Iterator

import pandas as pd

from src.normalization import normalize_dataframe


REQUIRED_NORMALIZED = {"entity_id", "norm_name", "norm_address"}


def _token_set(value: object) -> set[str]:
    if not isinstance(value, str) or not value:
        return set()
    return set(value.split())


def _has_normalized_columns(path: str) -> bool:
    columns = set(pd.read_csv(path, sep="\t", nrows=0).columns)
    return REQUIRED_NORMALIZED.issubset(columns)


def _normalized_cache_path(source_path: str, cache_dir: str) -> str:
    source = os.path.abspath(source_path)
    stat = os.stat(source)
    fingerprint = hashlib.sha256(
        f"{source}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:16]
    stem = os.path.splitext(os.path.basename(source))[0]
    return os.path.join(cache_dir, f"{stem}.{fingerprint}.normalized.tsv")


def ensure_normalized_tsv(
    source_path: str,
    *,
    cache_dir: str,
    chunksize: int = 200_000,
) -> str:
    """Use an already-normalized TSV or create/reuse a chunked disk cache."""
    if _has_normalized_columns(source_path):
        print(f"[rarity] Normalized columns found in {source_path}; using it directly", flush=True)
        return source_path

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = _normalized_cache_path(source_path, cache_dir)
    if os.path.isfile(cache_path) and _has_normalized_columns(cache_path):
        print(f"[rarity] Reusing normalized cache {cache_path}", flush=True)
        return cache_path

    print(f"[rarity] Creating normalized cache for {source_path}", flush=True)
    fd, temp_path = tempfile.mkstemp(prefix="normalizing_", suffix=".tsv", dir=cache_dir)
    os.close(fd)
    total = 0
    try:
        first = True
        for chunk_no, chunk in enumerate(pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=chunksize,
        ), start=1):
            chunk = normalize_dataframe(chunk, verbose=True)
            chunk.to_csv(
                temp_path,
                sep="\t",
                index=False,
                mode="w" if first else "a",
                header=first,
                encoding="utf-8",
            )
            total += len(chunk)
            first = False
            print(f"[rarity] Cached {total:,} rows from {source_path} (chunk {chunk_no})", flush=True)
        if first:
            raise ValueError(f"Input TSV is empty: {source_path}")
        os.replace(temp_path, cache_path)
    except BaseException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    print(f"[rarity] Normalized cache ready: {cache_path} ({total:,} rows)", flush=True)
    return cache_path


def _frames_from_tsv(path: str, chunksize: int) -> Iterator[pd.DataFrame]:
    for frame in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id", "norm_name", "norm_address"],
        chunksize=chunksize,
    ):
        yield frame


def _frames_from_memory(frames: tuple[pd.DataFrame, ...]) -> Iterator[pd.DataFrame]:
    for frame in frames:
        yield frame.loc[:, ["entity_id", "norm_name", "norm_address"]]


class _RarityIndex:
    def __init__(
        self,
        target_frames: Callable[[], Iterable[pd.DataFrame]],
        *,
        max_document_frequency: float,
        verbose: bool,
    ) -> None:
        if not 0 < max_document_frequency <= 1:
            raise ValueError("max_document_frequency must be in (0, 1]")
        self.target_ids: list[str] = []
        self.name_totals = array("f")
        self.address_totals = array("f")
        self.name_postings: dict[str, array] = {}
        self.address_postings: dict[str, array] = {}
        self.name_idf: dict[str, float] = {}
        self.address_idf: dict[str, float] = {}

        name_df: Counter[str] = Counter()
        address_df: Counter[str] = Counter()
        target_count = 0
        if verbose:
            print("[rarity] Pass 1/2: counting target token frequencies", flush=True)
        for frame in target_frames():
            for name, address in zip(frame["norm_name"], frame["norm_address"]):
                name_df.update(_token_set(name))
                address_df.update(_token_set(address))
            target_count += len(frame)
            if verbose:
                print(f"[rarity] Frequency pass: {target_count:,} target rows", flush=True)

        if target_count > 0xFFFFFFFF:
            raise ValueError("Compact postings support at most 4,294,967,295 target rows")
        max_df = max(1, int(target_count * max_document_frequency))
        self.name_idf = {
            token: math.log((target_count + 1) / (df + 1)) + 1.0
            for token, df in name_df.items() if df <= max_df
        }
        self.address_idf = {
            token: math.log((target_count + 1) / (df + 1)) + 1.0
            for token, df in address_df.items() if df <= max_df
        }
        del name_df, address_df

        if verbose:
            print(
                f"[rarity] Pass 2/2: building compact postings for {target_count:,} targets "
                f"(ignoring tokens in more than {max_df:,} records)",
                flush=True,
            )
        row_id = 0
        for frame in target_frames():
            for entity_id, name, address in zip(
                frame["entity_id"], frame["norm_name"], frame["norm_address"]
            ):
                name_tokens = _token_set(name).intersection(self.name_idf)
                address_tokens = _token_set(address).intersection(self.address_idf)
                name_weight = 0.0
                address_weight = 0.0
                for token in name_tokens:
                    weight = self.name_idf[token]
                    self.name_postings.setdefault(token, array("I")).append(row_id)
                    name_weight += weight
                for token in address_tokens:
                    weight = self.address_idf[token]
                    self.address_postings.setdefault(token, array("I")).append(row_id)
                    address_weight += weight
                self.target_ids.append(str(entity_id))
                self.name_totals.append(name_weight)
                self.address_totals.append(address_weight)
                row_id += 1
            if verbose:
                print(f"[rarity] Postings built: {row_id:,}/{target_count:,} targets", flush=True)

        if verbose:
            print(
                f"[rarity] Index ready: {len(self.name_postings):,} name tokens, "
                f"{len(self.address_postings):,} address tokens",
                flush=True,
            )

    @staticmethod
    def _rank_field(
        query_tokens: set[str],
        idf: dict[str, float],
        postings: dict[str, array],
        target_totals: array,
        *,
        threshold: float,
        top_k: int,
    ) -> list[tuple[int, float]]:
        usable = query_tokens.intersection(idf)
        query_total = sum(idf[token] for token in usable)
        if not usable or query_total <= 0:
            return []

        shared_weight: dict[int, float] = defaultdict(float)
        shared_count: dict[int, int] = defaultdict(int)
        for token in usable:
            weight = idf[token]
            for target_row in postings.get(token, ()):
                row = int(target_row)
                shared_weight[row] += weight
                shared_count[row] += 1

        ranked: list[tuple[int, float]] = []
        for row, common_weight in shared_weight.items():
            union_weight = query_total + target_totals[row] - common_weight
            score = common_weight / union_weight if union_weight > 0 else 0.0
            if shared_count[row] >= 1 and score >= threshold:
                ranked.append((row, score))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]

    def candidates(
        self,
        name: object,
        address: object,
        *,
        name_threshold: float,
        address_threshold: float,
        top_k_per_field: int,
        max_candidates: int,
    ) -> dict[str, float]:
        scores: dict[int, float] = {}
        for row, score in self._rank_field(
            _token_set(name), self.name_idf, self.name_postings, self.name_totals,
            threshold=name_threshold, top_k=top_k_per_field,
        ):
            scores[row] = max(scores.get(row, 0.0), score)
        for row, score in self._rank_field(
            _token_set(address), self.address_idf, self.address_postings, self.address_totals,
            threshold=address_threshold, top_k=top_k_per_field,
        ):
            scores[row] = max(scores.get(row, 0.0), score)

        best: dict[str, float] = {}
        for row, score in scores.items():
            target_id = self.target_ids[row]
            best[target_id] = max(best.get(target_id, 0.0), score)
        return dict(sorted(best.items(), key=lambda item: (-item[1], item[0]))[:max_candidates])


def generate_rarity_from_tsv(
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
    """Create a submission-format candidate TSV with a cached normalized input."""
    if chunksize < 1 or top_k_per_field < 1 or max_candidates < 1:
        raise ValueError("chunksize, top_k_per_field, and max_candidates must be positive")
    if not 0 <= name_threshold <= 1 or not 0 <= address_threshold <= 1:
        raise ValueError("name_threshold and address_threshold must be between 0 and 1")
    cache_dir = os.path.abspath(cache_dir)
    source1_norm = ensure_normalized_tsv(source1_path, cache_dir=cache_dir, chunksize=chunksize)
    source2_norm = ensure_normalized_tsv(source2_path, cache_dir=cache_dir, chunksize=chunksize)
    source3_norm = ensure_normalized_tsv(source3_path, cache_dir=cache_dir, chunksize=chunksize)
    target_factory = lambda: chain(
        _frames_from_tsv(source2_norm, chunksize),
        _frames_from_tsv(source3_norm, chunksize),
    )
    index = _RarityIndex(
        target_factory,
        max_document_frequency=max_document_frequency,
        verbose=True,
    )

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    fd, temp_output = tempfile.mkstemp(prefix="candidate_pairs_", suffix=".tsv", dir=output_dir)
    os.close(fd)
    query_count = 0
    try:
        with open(temp_output, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", "candidate_entity_ids"])
            for frame in _frames_from_tsv(source1_norm, chunksize):
                for entity_id, name, address in zip(
                    frame["entity_id"], frame["norm_name"], frame["norm_address"]
                ):
                    scored = index.candidates(
                        name, address,
                        name_threshold=name_threshold,
                        address_threshold=address_threshold,
                        top_k_per_field=top_k_per_field,
                        max_candidates=max_candidates,
                    )
                    writer.writerow([str(entity_id), ",".join(scored)])
                    query_count += 1
                    if query_count % 200_000 == 0:
                        print(f"[rarity] Candidate generation: {query_count:,} Source 1 rows", flush=True)
        os.replace(temp_output, output_path)
    except BaseException:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        raise
    print(f"[rarity] Wrote candidates for {query_count:,} Source 1 records to {output_path}", flush=True)
    return query_count


def generate_rarity_from_frames(
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
    """In-memory DataFrame API, principally for smaller programmatic inputs."""
    if top_k_per_field < 1 or max_candidates < 1:
        raise ValueError("top_k_per_field and max_candidates must be positive")
    if not 0 <= name_threshold <= 1 or not 0 <= address_threshold <= 1:
        raise ValueError("name_threshold and address_threshold must be between 0 and 1")
    from src.blocking import _prepare_frame

    s1 = _prepare_frame(source1, "Source 1").reset_index(drop=True)
    s2 = _prepare_frame(source2, "Source 2").reset_index(drop=True)
    s3 = _prepare_frame(source3, "Source 3").reset_index(drop=True)
    target_frames = (s2, s3)
    index = _RarityIndex(
        lambda: _frames_from_memory(target_frames),
        max_document_frequency=max_document_frequency,
        verbose=True,
    )
    rows: list[dict[str, str]] = []
    for position, record in enumerate(s1.itertuples(index=False), start=1):
        scored = index.candidates(
            record.norm_name, record.norm_address,
            name_threshold=name_threshold,
            address_threshold=address_threshold,
            top_k_per_field=top_k_per_field,
            max_candidates=max_candidates,
        )
        rows.append({"source1_entity_id": str(record.entity_id), "candidate_entity_ids": ",".join(scored)})
        if position % 200_000 == 0 or position == len(s1):
            print(f"[rarity] Candidate generation: {position:,}/{len(s1):,} Source 1 rows", flush=True)
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
