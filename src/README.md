# Business Entity Resolution Pipeline

This repository contains the end-to-end Machine Learning pipeline for the Business Entity Resolution Challenge.

## Pipeline Architecture
1. **Normalization (`src/normalization.py`)**: Canonicalizing business names and addresses (Unicode transliteration, legal suffix stripping, abbreviation expansion, ordinal/acronym standardisation).
2. **Candidate Generation / Blocking (`src/blocking.py`)**: Rarity-weighted name/address token retrieval.
3. **Feature Engineering (`src/features.py`)**: Similarity metrics computation (Levenshtein, Jaccard, TF-IDF cosine, length deltas, exact matches).
4. **Matching Model (`src/model.py`)**: High-precision ML classifier with macro F_0.5 threshold optimization.
5. **Output Generator (`src/inference.py`)**: Generation and validation of `matching_results.tsv` and `candidate_pairs.tsv`.

## Normalization Usage
```python
from src.normalization import normalize_name, normalize_address

name = normalize_name("Shree Krishna Medical Store Pvt. Ltd.")
# -> "shree krishna medical store"

addr = normalize_address("12, M.G. Road, Anand")
# -> "12 mg road anand"
```

## Candidate Generation Usage

The default in `src/blocking.py` is the rarity-weighted inverted index. It
counts token document frequencies across Sources 2 and 3, gives rarer name and
address tokens more weight, retrieves and ranks candidates from each field,
then unions the lists. Frequent tokens above the configured document-frequency
cap are skipped to keep lookups bounded. Country is not used for blocking.

For TSV input, it normalizes in chunks and saves normalized TSV caches under
`data/normalized_cache`. Cache names include the source path, file size, and
modification time, so later runs reuse normalized columns when the input has
not changed. The inverted index uses compact integer postings and streams the
cached files rather than loading all target records into pandas at once.

Run rarity-weighted blocking from the repository root:

```cmd
venv\Scripts\python.exe -m src.blocking --source1 dataset\student_resource\dataset\train\train_source1.tsv --source2 dataset\student_resource\dataset\train\train_source2.tsv --source3 dataset\student_resource\dataset\train\train_source3.tsv --output output\train_candidate_pairs.tsv
```

Use the corresponding `dataset\student_resource\dataset\test\test_source*.tsv`
paths and `output\candidate_pairs.tsv` to generate test candidates. Defaults
are `--max-document-frequency 0.02`, `--name-threshold 0.18`,
`--address-threshold 0.18`, `--top-k-per-field 50`, and `--max-candidates 100`.
Evaluate blocking on a deterministic held-out portion of the labeled training
queries before generating test candidates:

```cmd
venv\Scripts\python.exe -m src.evaluate_blocking --holdout-fraction 0.20 --seed 42
```

The report includes ground-truth edge recall, positive-query recall, MRR using
the candidate retrieval order, average candidates, reduction ratio, and the
share of singleton queries that receive any candidates. Tune the rarity
frequency cap, score thresholds, and candidate caps against this held-out report.
