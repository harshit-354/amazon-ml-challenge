# Business Entity Resolution Pipeline

This repository contains the end-to-end Machine Learning pipeline for the Amazon Business Entity Resolution Challenge.

## Pipeline Architecture
1. **Normalization (`src/normalization.py`)**: Canonicalizing business names and addresses (Unicode transliteration, legal suffix stripping, abbreviation expansion, ordinal/acronym standardisation).
2. **Candidate Generation / Blocking (`src/blocking.py`, `src/rarity_blocking.py`)**: Rarity-weighted name and address inverted indexing with sub-quadratic search space pruning.
3. **Feature Engineering (`src/features.py`)**: Multi-metric similarity computation:
   - Business name: Levenshtein, Jaccard, character 3-gram TF-IDF cosine similarity.
   - Business address: Levenshtein, Jaccard, character 3-gram TF-IDF cosine similarity.
   - Auxiliary features: common token counts, string length differences, exact match indicators, and token overlap ratios.
4. **Matching Model (`src/model.py`)**: Supervised tree-based classifier (`HistGradientBoostingClassifier`) with hard negative mining, calibrated probability predictions, and decision threshold optimization for macro $F_{0.5}$.
5. **Output Inference & Validation (`src/inference.py`)**: Streaming candidate pair evaluation, singleton handling, `output/matching_results.tsv` generation, and automated validation via `student_resource/utils/validate_submission.py`.

---

## 1. Normalization Usage
```python
from src.normalization import normalize_name, normalize_address

name = normalize_name("Shree Krishna Medical Store Pvt. Ltd.")
# -> "shree krishna medical store"

addr = normalize_address("12, M.G. Road, Anand")
# -> "12 mg road anand"
```

---

## 2. Candidate Generation / Blocking
Run rarity-weighted blocking from the repository root:

```bash
python3 -m src.blocking \
  --source1 student_resource/dataset/test/test_source1.tsv \
  --source2 student_resource/dataset/test/test_source2.tsv \
  --source3 student_resource/dataset/test/test_source3.tsv \
  --output output/candidate_pairs.tsv
```

---

## 3. Matching Model Training & Validation
Train the classifier on labeled training pairs with hard negative mining and optimize the decision threshold for macro $F_{0.5}$:

```bash
python3 -m src.model \
  --train-dir student_resource/dataset/train \
  --num-queries 25000 \
  --val-fraction 0.20 \
  --output-model models/matching_model.pkl
```

---

## 4. Inference & Submission Generation
Score candidates from `candidate_pairs.tsv` and produce `output/matching_results.tsv`:

```bash
python3 -m src.inference \
  --candidate-pairs output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test \
  --model models/matching_model.pkl \
  --output output/matching_results.tsv
```

---

## 5. Submission Validation
Verify submission files using the competition validator:

```bash
python3 student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```
