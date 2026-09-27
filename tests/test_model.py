"""Unit tests for matching model, F0.5 macro evaluation, and threshold optimization."""

import unittest
from sklearn.ensemble import HistGradientBoostingClassifier
from src.model import compute_entity_f05, evaluate_macro_f05, optimize_threshold


class TestModel(unittest.TestCase):
    def test_singleton_scoring(self):
        # Correctly predicting empty for a singleton scores 1.0
        self.assertEqual(compute_entity_f05(set(), set()), 1.0)
        # Any false positive for a singleton scores 0.0
        self.assertEqual(compute_entity_f05(set(), {"S2-100"}), 0.0)

    def test_example_from_problem_statement(self):
        # Ground truth: {S2-00047, S3-00812}
        # Pred: {S2-00047, S2-00193, S3-00812}
        # F0.5 should be 0.714
        true_set = {"S2-00047", "S3-00812"}
        pred_set = {"S2-00047", "S2-00193", "S3-00812"}
        score = compute_entity_f05(true_set, pred_set)
        self.assertAlmostEqual(score, 0.714, places=3)

    def test_macro_evaluation(self):
        gt = {
            "S1-1": {"S2-A", "S3-B"},
            "S1-2": set(),  # singleton
            "S1-3": {"S2-C"},
        }
        # S1-1: exact match -> 1.0
        # S1-2: empty pred -> 1.0
        # S1-3: missed -> 0.0
        preds = {
            "S1-1": {"S2-A", "S3-B"},
            "S1-2": set(),
            "S1-3": set(),
        }
        metrics = evaluate_macro_f05(gt, preds)
        # Macro average = (1.0 + 1.0 + 0.0) / 3 = 0.6667
        self.assertAlmostEqual(metrics["macro_f05"], 2.0 / 3.0, places=4)
        self.assertEqual(metrics["singleton_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
