import numpy as np
import pandas as pd
import os
import sys

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.abspath("."))

def make_stratified_folds(y, k=5, random_state=42):
    """Create stratified test_index folds manually."""
    y = pd.Series(y).reset_index(drop=True)
    rng = np.random.RandomState(random_state)

    class_0_idx = np.where(y == 0)[0]
    class_1_idx = np.where(y == 1)[0]

    if len(class_0_idx) < k or len(class_1_idx) < k:
        raise ValueError("Each calss must have at least k samples for stratified k-fold.")
    
    rng.shuffle(class_0_idx)
    rng.shuffle(class_1_idx)

    class_0_folds = np.array_split(class_0_idx, k)
    class_1_folds = np.array_split(class_1_idx, k)

    folds = []
    for i in range(k):
        fold_idx = np.concatenate([class_0_folds[i], class_1_folds[i]])
        rng.shuffle(fold_idx)
        folds.append(fold_idx)

    return folds

def custom_stratified_cv_scores(X, y, pipeline, k=5, random_state=42):
    """Run custom stratified k-fold cross-validation manually."""
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    folds = make_stratified_folds(y, k=k, random_state=random_state)

    scores = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    all_indices = np.arange(len(y))

    for test_idx in folds:
        train_idx = np.setdiff1d(all_indices, test_idx)

        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]

        model = clone(pipeline)
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)

        scores["accuracy"].append(accuracy_score(y_test_fold, y_pred))
        scores["precision"].append(precision_score(y_test_fold, y_pred, zero_division=0))
        scores["recall"].append(recall_score(y_test_fold, y_pred, zero_division=0))
        scores["f1"].append(f1_score(y_test_fold, y_pred, zero_division=0))

    return scores
    
def summarize_custom_cv(scores):
    """Print mean and std for custom CV scores."""
    print("\n=== Tier 3: Custom Stratified CV Results ===")
    for metric, values in scores.items():
        print(f"{metric}: mean={np.mean(values):.6f}, std={np.std(values):.6f}, folds={values}")

def compare_custom_vs_sklearn(X, y, pipeline, k=5, random_state=42):
    """Compare custom stratified CV scores with sklearn cross_validate."""
    custom_scores = custom_stratified_cv_scores(X, y, pipeline, k=k, random_state=random_state)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    sklearn_scores = cross_validate(
        pipeline,
        X,
        y,
        cv=skf,
        scoring=["accuracy", "precision", "recall", "f1"]
    )

    print("\n=== Tier 3: Custom vs sklearn ===")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        custom_mean = np.mean(custom_scores[metric])
        sklearn_mean = np.mean(sklearn_scores[f"test_{metric}"])
        diff = custom_mean - sklearn_mean

        print(
            f"{metric}: custom_mean={custom_mean:.6f}, "
            f"sklearn_mean={sklearn_mean:.6f}, difference={diff:.6f}"
        )

    return custom_scores, sklearn_scores