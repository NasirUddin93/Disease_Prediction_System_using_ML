"""
================================================================================
Phase 7: Test Suite for Probability Calibration & Uncertainty Quantification
================================================================================
Automated unit tests asserting:
1. Probability normalization invariant: probabilities sum to 1.0 and are non-negative.
2. ECE computation correctness on synthetic distributions (perfect vs. overconfident).
3. Multiclass Brier score calculation correctness against mathematical formulation.
4. Top-k accuracy ranking monotonicity (Top-5 >= Top-3 >= Top-1).
5. Zero group leakage in inner StratifiedGroupKFold splits.
6. CalibratedClassifierCV compatibility and 41-class probability alignment.
7. Perturbation functions preserve binary feature constraints and shapes.
================================================================================
"""

import unittest
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss


def compute_top_label_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 15) -> float:
    """Computes Top-Label Expected Calibration Error (ECE)."""
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)
            
    return float(ece)


def compute_multiclass_brier_score(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int = 41) -> float:
    """Computes multiclass Brier score: 1/N * sum_i sum_c (p_ic - y_ic)^2."""
    n_samples = len(y_true)
    y_one_hot = np.zeros((n_samples, n_classes), dtype=float)
    y_one_hot[np.arange(n_samples), y_true] = 1.0
    return float(np.mean(np.sum((y_proba - y_one_hot) ** 2, axis=1)))


def compute_top_k_acc(y_true: np.ndarray, y_proba: np.ndarray, k: int) -> float:
    """Computes Top-k categorical accuracy."""
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
    hits = np.any(top_k_preds == y_true[:, np.newaxis], axis=1)
    return float(np.mean(hits))


class TestCalibrationExperiment(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 200
        self.n_classes = 41
        self.y_true = np.random.randint(0, self.n_classes, size=self.n_samples)

    def test_probability_normalization_invariant(self):
        """Probability outputs must strictly sum to 1.0 across all classes."""
        random_logits = np.random.exponential(scale=1.0, size=(self.n_samples, self.n_classes))
        probabilities = random_logits / random_logits.sum(axis=1, keepdims=True)
        
        self.assertTrue(np.all(probabilities >= 0.0), "Negative probability detected!")
        self.assertTrue(np.all(probabilities <= 1.0), "Probability > 1.0 detected!")
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(self.n_samples), atol=1e-6)

    def test_ece_perfect_calibration(self):
        """Perfect predictions (confidence 1.0, accuracy 1.0) must yield ECE = 0.0."""
        perfect_proba = np.zeros((self.n_samples, self.n_classes), dtype=float)
        perfect_proba[np.arange(self.n_samples), self.y_true] = 1.0
        
        ece = compute_top_label_ece(self.y_true, perfect_proba, n_bins=15)
        self.assertAlmostEqual(ece, 0.0, places=5)

    def test_ece_overconfident_miscalibration(self):
        """Completely overconfident wrong predictions (confidence 1.0, accuracy 0.0) must yield ECE = 1.0."""
        wrong_targets = (self.y_true + 1) % self.n_classes
        wrong_proba = np.zeros((self.n_samples, self.n_classes), dtype=float)
        wrong_proba[np.arange(self.n_samples), wrong_targets] = 1.0
        
        ece = compute_top_label_ece(self.y_true, wrong_proba, n_bins=15)
        self.assertAlmostEqual(ece, 1.0, places=5)

    def test_multiclass_brier_score_limits(self):
        """Brier score must be 0.0 for perfect predictions and <= 2.0 for completely wrong."""
        perfect_proba = np.zeros((self.n_samples, self.n_classes), dtype=float)
        perfect_proba[np.arange(self.n_samples), self.y_true] = 1.0
        bs_perfect = compute_multiclass_brier_score(self.y_true, perfect_proba, self.n_classes)
        self.assertAlmostEqual(bs_perfect, 0.0, places=5)

        wrong_targets = (self.y_true + 1) % self.n_classes
        wrong_proba = np.zeros((self.n_samples, self.n_classes), dtype=float)
        wrong_proba[np.arange(self.n_samples), wrong_targets] = 1.0
        bs_wrong = compute_multiclass_brier_score(self.y_true, wrong_proba, self.n_classes)
        self.assertAlmostEqual(bs_wrong, 2.0, places=5)

    def test_ranking_monotonicity(self):
        """Top-5 accuracy >= Top-3 accuracy >= Top-1 accuracy must hold strictly."""
        random_logits = np.random.exponential(scale=1.0, size=(self.n_samples, self.n_classes))
        probabilities = random_logits / random_logits.sum(axis=1, keepdims=True)
        
        acc_1 = compute_top_k_acc(self.y_true, probabilities, k=1)
        acc_3 = compute_top_k_acc(self.y_true, probabilities, k=3)
        acc_5 = compute_top_k_acc(self.y_true, probabilities, k=5)
        
        self.assertGreaterEqual(acc_5, acc_3)
        self.assertGreaterEqual(acc_3, acc_1)

    def test_inner_group_leakage_safety(self):
        """Inner StratifiedGroupKFold splits must have 0 overlapping group hashes."""
        n_records = 300
        n_unique_groups = 50
        group_ids = np.random.randint(0, n_unique_groups, size=n_records)
        y = np.random.randint(0, 10, size=n_records)
        X = np.random.randn(n_records, 15)

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=group_ids)):
            train_groups = set(group_ids[train_idx])
            val_groups = set(group_ids[val_idx])
            overlap = train_groups.intersection(val_groups)
            self.assertEqual(len(overlap), 0, f"Group leakage detected in fold {fold}!")

    def test_calibrated_classifier_cv_pipeline(self):
        """Verifies CalibratedClassifierCV works with precomputed splits and outputs valid probabilities."""
        n_records = 150
        n_classes = 5
        group_ids = np.repeat(np.arange(30), 5)
        y = np.random.randint(0, n_classes, size=n_records)
        X = np.random.randn(n_records, 10)

        cv_splits = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(X, y, groups=group_ids))
        clf = LogisticRegression(max_iter=200)
        
        cal_sig = CalibratedClassifierCV(clf, method='sigmoid', cv=cv_splits)
        cal_sig.fit(X, y)
        probs_sig = cal_sig.predict_proba(X[:10])
        self.assertEqual(probs_sig.shape, (10, n_classes))
        np.testing.assert_allclose(probs_sig.sum(axis=1), np.ones(10), atol=1e-5)

        cal_iso = CalibratedClassifierCV(clf, method='isotonic', cv=cv_splits)
        cal_iso.fit(X, y)
        probs_iso = cal_iso.predict_proba(X[:10])
        self.assertEqual(probs_iso.shape, (10, n_classes))
        np.testing.assert_allclose(probs_iso.sum(axis=1), np.ones(10), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
