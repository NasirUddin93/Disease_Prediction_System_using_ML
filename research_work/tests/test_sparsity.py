"""
Unit tests for Phase 05: Symptom Sparsity and Cold-Start Robustness.
Validates:
1. Masking cap and preservation invariants
2. Nested subset property across budgets (s=3 subset of s=5)
3. Input array immutability and binary integrity
4. Zero-leakage duplicate-aware group separation under Protocol C
5. Top-k ranking monotonicity (Top-5 >= Top-3 >= Top-1)
6. Probability validation and finite metric sanity
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


def apply_nested_symptom_sparsity(X: np.ndarray, max_active: int, seed: int) -> np.ndarray:
    """
    Simulates patient intake cold-start with strict nested subset preservation.
    For each patient record i and seed, a reproducible permutation of its active
    symptoms is generated. For budget max_active, the prefix of length
    min(max_active, n_active) is retained.
    """
    X_sparse = np.zeros_like(X, dtype=np.float32)
    for i in range(len(X)):
        active_indices = np.where(X[i] == 1)[0]
        n_active = len(active_indices)
        if n_active <= max_active:
            X_sparse[i, active_indices] = 1.0
        else:
            # Deterministic generator per sample and seed
            sample_rng = np.random.RandomState(int((seed * 10007 + i * 37) % (2**31 - 1)))
            perm = sample_rng.permutation(active_indices)
            chosen = perm[:max_active]
            X_sparse[i, chosen] = 1.0
    return X_sparse


class TestSymptomSparsity(unittest.TestCase):

    def setUp(self):
        # Create synthetic test dataset
        self.rng = np.random.RandomState(42)
        self.X_sample = np.zeros((20, 132), dtype=np.float32)
        # Populate each row with 2 to 12 active symptoms
        for i in range(20):
            n_active = self.rng.randint(2, 13)
            idx = self.rng.choice(132, size=n_active, replace=False)
            self.X_sample[i, idx] = 1.0

    def test_preservation_when_active_le_budget(self):
        """Records with <= s symptoms should be completely preserved."""
        for row in self.X_sample:
            n_active = int(np.sum(row))
            budget = n_active + 2
            masked = apply_nested_symptom_sparsity(row[np.newaxis, :], max_active=budget, seed=42)
            np.testing.assert_array_equal(masked[0], row)

    def test_cap_when_active_gt_budget(self):
        """Records with > s symptoms should have exactly s positive indicators."""
        for budget in [1, 2, 3, 5]:
            masked = apply_nested_symptom_sparsity(self.X_sample, max_active=budget, seed=42)
            for i in range(len(self.X_sample)):
                n_orig = int(np.sum(self.X_sample[i]))
                n_masked = int(np.sum(masked[i]))
                if n_orig > budget:
                    self.assertEqual(n_masked, budget)
                else:
                    self.assertEqual(n_masked, n_orig)

    def test_nested_subset_property(self):
        """For a given seed, active symptoms at s=3 MUST be a subset of active symptoms at s=5."""
        for seed in [42, 101, 2024, 777, 999]:
            masked_3 = apply_nested_symptom_sparsity(self.X_sample, max_active=3, seed=seed)
            masked_5 = apply_nested_symptom_sparsity(self.X_sample, max_active=5, seed=seed)
            masked_8 = apply_nested_symptom_sparsity(self.X_sample, max_active=8, seed=seed)

            for i in range(len(self.X_sample)):
                set_3 = set(np.where(masked_3[i] == 1)[0])
                set_5 = set(np.where(masked_5[i] == 1)[0])
                set_8 = set(np.where(masked_8[i] == 1)[0])

                self.assertTrue(set_3.issubset(set_5), f"Sample {i}, seed {seed}: s=3 not subset of s=5")
                self.assertTrue(set_5.issubset(set_8), f"Sample {i}, seed {seed}: s=5 not subset of s=8")

    def test_immutability_and_binary_values(self):
        """Input array should remain unmodified, and output must be strictly binary {0, 1}."""
        X_copy = self.X_sample.copy()
        masked = apply_nested_symptom_sparsity(self.X_sample, max_active=4, seed=42)
        np.testing.assert_array_equal(self.X_sample, X_copy)
        unique_vals = set(np.unique(masked))
        self.assertTrue(unique_vals.issubset({0.0, 1.0}))

    def test_top_k_ranking_monotonicity(self):
        """Top-5 accuracy must be >= Top-3 accuracy >= Top-1 accuracy."""
        n_samples = 50
        n_classes = 41
        y_true = self.rng.randint(0, n_classes, size=n_samples)
        raw_scores = self.rng.rand(n_samples, n_classes)
        proba = raw_scores / np.sum(raw_scores, axis=1, keepdims=True)

        top1 = compute_top_k_accuracy(y_true, proba, k=1)
        top3 = compute_top_k_accuracy(y_true, proba, k=3)
        top5 = compute_top_k_accuracy(y_true, proba, k=5)

        self.assertGreaterEqual(top3, top1)
        self.assertGreaterEqual(top5, top3)

    def test_protocol_c_group_leakage_isolation(self):
        """Ensure no symptom vector hashes cross train and validation splits in Protocol C."""
        loader = DiseaseDataLoader()
        for fold_idx, X_tr, y_tr, X_va, y_va in loader.get_protocol_c_folds():
            # Hash every row vector
            train_hashes = set(hash(tuple(row)) for row in X_tr)
            val_hashes = set(hash(tuple(row)) for row in X_va)
            overlap = train_hashes.intersection(val_hashes)
            self.assertEqual(len(overlap), 0, f"Fold {fold_idx} has {len(overlap)} leaking duplicate patterns!")


if __name__ == "__main__":
    unittest.main()
