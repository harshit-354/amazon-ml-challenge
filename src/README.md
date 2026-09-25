# Business Entity Resolution Pipeline

This repository contains the end-to-end Machine Learning pipeline for the Business Entity Resolution Challenge.

## Pipeline Architecture
1. **Normalization (`src/normalization.py`)**: Canonicalizing business names and addresses (Unicode transliteration, legal suffix stripping, abbreviation expansion, ordinal/acronym standardisation).
2. **Candidate Generation / Blocking (`src/blocking.py`)**: Multi-method candidate pair generation using token overlap and similarity blocking.
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

`src/blocking.py` unions three indexed retrieval passes: normalized-name token
Jaccard, normalized-address token Jaccard, and character-trigram Jaccard on
names. Country is not used for blocking. The output has one row for every
Source 1 record, including an empty candidate list when no pass finds a result.

Run it from the repository root with the challenge TSV paths:

```bash
python -m src.blocking \
  --source1 src/dataset/student_resource/dataset/train/train_source1.tsv \
  --source2 src/dataset/student_resource/dataset/train/train_source2.tsv \
  --source3 src/dataset/student_resource/dataset/train/train_source3.tsv \
  --output output/train_candidate_pairs.tsv
```

Use the corresponding `test/test_source*.tsv` paths to create test candidates.
Thresholds and the per-method candidate cap can be adjusted with
`--token-threshold`, `--char-threshold`, and `--max-per-method`. The defaults
are starting points; tune them on a held-out training split by measuring
blocking recall and candidate volume before training the matching model.
