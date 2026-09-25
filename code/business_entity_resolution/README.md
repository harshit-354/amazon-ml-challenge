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
