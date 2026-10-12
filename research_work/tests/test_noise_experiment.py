"""
Unit tests for Phase 06: Stochastic Symptom-Reporting Noise Benchmark.
Validates:
1. False-negative-only mutations (1 -> 0 only)
2. False-positive-only mutations (0 -> 1 only)
3. Combined noise binary integrity ({0, 1} output only)
4. Nested monotonicity (increasing error rates strictly expand mutations)
5. Input array immutability
6. Zero-leakage duplicate-aware group separation under Protocol C
7. Top-k ranking monotonicity (Top-5 >= Top-3 >= Top-1)
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(SRC_DIR))

from evaluation import compute_top_k_accuracy
from data_loader import DiseaseDataLoader


def apply_reporting_noise(
    X: np.ndarray,
    dropout_uniforms: np.ndarray,
    false_positive_uniforms: np.ndarray,
    p_dropout: float = 0.0,
    p_false_positive: float = 0.0,
) -> np.ndarray:
    """
    Injects stochastic reporting noise into binary symptom vectors.
    1 -> 0 (Dropout): Present symptom omitted if uniform < p_dropout.
    0 -> 1 (Activation): Absent symptom activated if uniform < p_false_positive.
    """
    noisy = X.copy()
    present_mask = (X == 1)
    absent_mask = (X == 0)

    if p_dropout > 0.0:
        dropped = present_mask & (dropout_uniforms < p_dropout)
        noisy[dropped] = 0.0

    if p_false_positive > 0.0:
        activated = absent_mask & (false_positive_uniforms < p_false_positive)
        noisy[activated] = 1.0

    return noisy


class TestNoiseExperiment(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.X_sample = np.zeros((30, 132), dtype=np.float32)
        for i in range(30):
            n_active = self.rng.randint(3, 12)
            idx = self.rng.choice(132, size=n_active, replace=False)
            self.X_sample[i, idx] = 1.0

        self.u_drop = self.rng.uniform(0.0, 1.0, size=self.X_sample.shape)
        self.u_fp = self.rng.uniform(0.0, 1.0, size=self.X_sample.shape)

    def test_false_negative_only(self):
        """When p_fp=0, only 1s can become 0; no 0 can become 1."""
        noisy = apply_reporting_noise(self.X_sample, self.u_drop, self.u_fp, p_dropout=0.20, p_false_positive=0.0)
        # Check that original 0s remained 0
        orig_zeros = (self.X_sample == 0)
        self.assertTrue(np.all(noisy[orig_zeros] == 0.0))
        # Total 1s in noisy should be <= total 1s in original
        self.assertLessEqual(np.sum(noisy), np.sum(self.X_sample))

    def test_false_positive_only(self):
        """When p_drop=0, only 0s can become 1; no 1 can become 0."""
        noisy = apply_reporting_noise(self.X_sample, self.u_drop, self.u_fp, p_dropout=0.0, p_false_positive=0.05)
        # Check that original 1s remained 1
        orig_ones = (self.X_sample == 1)
        self.assertTrue(np.all(noisy[orig_ones] == 1.0))
        # Total 1s in noisy should be >= total 1s in original
        self.assertGreaterEqual(np.sum(noisy), np.sum(self.X_sample))

    def test_combined_noise_binary_integrity(self):
        """Combined noise must preserve strictly binary values {0, 1}."""
        noisy = apply_reporting_noise(self.X_sample, self.u_drop, self.u_fp, p_dropout=0.15, p_false_positive=0.05)
        unique_vals = set(np.unique(noisy))
        self.assertTrue(unique_vals.issubset({0.0, 1.0}))

    def test_nested_monotonicity(self):
        """Increasing p_dropout strictly expands dropped symptoms for fixed uniforms."""
        noisy_10 = apply_reporting_noise(self.X_sample, self.u_drop, self.u_fp, p_dropout=0.10, p_false_positive=0.0)
        noisy_20 = apply_reporting_noise(self.X_sample, self.u_drop, self.u_fp, p_dropout=0.20, p_false_positive=0.0)
        noisy_30 = apply_reporting_noise(self.X_sample, self.u_drop, self.u_fp, p_dropout=0.30, p_false_positive=0.0)

        for i in range(len(self.X_sample)):
            # Retained 1s at higher dropout must be a subset of retained 1s at lower dropout
            retained_10 = set(np.where(noisy_10[i] == 1)[0])
            retained_20 = set(np.where(noisy_20[i] == 1)[0])
            retained_30 = set(np.where(noisy_30[i] == 1)[0])

            self.assertTrue(retained_20.issubset(retained_10))
            self.assertTrue(retained_30.issubset(retained_20))

    def test_immutability(self):
        """Original input array must not be mutated in-place."""
        X_copy = self.X_sample.copy()
        _ = apply_reporting_noise(self.X_sample, self.u_drop, self.u_fp, p_dropout=0.20, p_false_positive=0.10)
        np.testing.assert_array_equal(self.X_sample, X_copy)

    def test_protocol_c_group_leakage_isolation(self):
        """Ensure no symptom vector hashes cross train and validation splits in Protocol C."""
        loader = DiseaseDataLoader()
        for fold_idx, X_tr, y_tr, X_va, y_va in loader.get_protocol_c_folds():
            train_hashes = set(hash(tuple(row)) for row in X_tr)
            val_hashes = set(hash(tuple(row)) for row in X_va)
            overlap = train_hashes.intersection(val_hashes)
            self.assertEqual(len(overlap), 0, f"Fold {fold_idx} has {len(overlap)} leaking duplicate patterns!")

    def test_top_k_ranking_monotonicity(self):
        """Top-5 accuracy must be >= Top-3 accuracy >= Top-1 accuracy."""
        n_samples = 40
        n_classes = 41
        y_true = self.rng.randint(0, n_classes, size=n_samples)
        raw_scores = self.rng.rand(n_samples, n_classes)
        proba = raw_scores / np.sum(raw_scores, axis=1, keepdims=True)

        top1 = compute_top_k_accuracy(y_true, proba, k=1)
        top3 = compute_top_k_accuracy(y_true, proba, k=3)
        top5 = compute_top_k_accuracy(y_true, proba, k=5)

        self.assertGreaterEqual(top3, top1)
        self.assertGreaterEqual(top5, top3)


if __name__ == "__main__":
    unittest.main()
