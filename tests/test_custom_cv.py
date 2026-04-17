import numpy as np
import pandas as pd
import pytest
import os
import sys

sys.path.insert(0, os.path.abspath("."))

from custom_cv import make_stratified_folds


def test_number_of_folds():
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    folds = make_stratified_folds(y, k=5, random_state=42)
    assert len(folds) == 5


def test_non_overlapping_test_sets():
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    folds = make_stratified_folds(y, k=5, random_state=42)

    all_test_indices = np.concatenate(folds)
    unique_indices = np.unique(all_test_indices)

    assert len(all_test_indices) == len(unique_indices)


def test_class_ratios_preserved():
    y = pd.Series([0] * 80 + [1] * 20)
    folds = make_stratified_folds(y, k=5, random_state=42)

    overall_ratio = (y == 1).mean()

    for fold in folds:
        fold_ratio = (y.iloc[fold] == 1).mean()
        assert abs(fold_ratio - overall_ratio) < 0.05


def test_raises_when_class_has_fewer_than_k_samples():
    y = pd.Series([0, 0, 0, 0, 1])
    with pytest.raises(ValueError):
        make_stratified_folds(y, k=3, random_state=42)