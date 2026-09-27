"""Unit tests for Feature Extraction module."""

import unittest
import numpy as np
from src.features import extract_pair_features, extract_batch_features, FEATURE_NAMES


class TestFeatures(unittest.TestCase):
    def test_feature_count_and_names(self):
        f = extract_pair_features("apple", "123 main st", "apple", "123 main st")
        self.assertEqual(len(f), len(FEATURE_NAMES))

    def test_identical_records(self):
        f = extract_pair_features("google inc", "1600 amphitheatre pkwy", "google inc", "1600 amphitheatre pkwy")
        # Levenshtein should be 1.0
        self.assertAlmostEqual(f[0], 1.0)
        # Jaccard should be 1.0
        self.assertAlmostEqual(f[1], 1.0)
        # Cosine should be 1.0
        self.assertAlmostEqual(f[2], 1.0)
        # Exact matches should be 1.0
        self.assertEqual(f[10], 1.0)
        self.assertEqual(f[11], 1.0)

    def test_completely_different_records(self):
        f = extract_pair_features("apple", "california", "starbucks", "seattle")
        # Levenshtein low
        self.assertLess(f[0], 0.3)
        # Jaccard 0.0
        self.assertAlmostEqual(f[1], 0.0)
        self.assertAlmostEqual(f[4], 0.0)
        # Exact matches 0.0
        self.assertEqual(f[10], 0.0)
        self.assertEqual(f[11], 0.0)

    def test_batch_extraction(self):
        s1_n = ["apple", "banana"]
        s1_a = ["1st st", "2nd ave"]
        c_n = ["apple inc", "banana store"]
        c_a = ["1st street", "2nd avenue"]
        batch = extract_batch_features(s1_n, s1_a, c_n, c_a)
        self.assertIsInstance(batch, np.ndarray)
        self.assertEqual(batch.shape, (2, len(FEATURE_NAMES)))


if __name__ == "__main__":
    unittest.main()
