"""
Module 5 Week A — Integration: ML Evaluation Pipeline

Build a structured evaluation pipeline that compares 5 model
configurations using cross-validation with ColumnTransformer + Pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, cross_val_predict, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")

NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents"]

CATEGORICAL_FEATURES = ["gender", "contract_type", "internet_service",
                        "payment_method"]


def load_and_prepare(filepath="data/telecom_churn.csv"):
    """Load data and separate features from target.

    Returns:
        Tuple of (X, y) where X is a DataFrame of features
        and y is a Series of the target (churned).
    """
    df = pd.read_csv(filepath)

    # Drop customer_id if it exists
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])
    
    X = df.drop(columns=["churned"])
    y = df["churned"]

    return X, y


def build_preprocessor():
    """Build a ColumnTransformer for numeric and categorical features.

    Returns:
        ColumnTransformer that scales numeric features and
        one-hot encodes categorical features.
    """
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES)
    ])

    return preprocessor


def define_models():
    """Define the 5 model configurations to compare.

    Two dummy baselines are included to teach two different lessons:
    most_frequent demonstrates the accuracy inflation problem on imbalanced
    data; stratified shows what random guessing in proportion to class
    frequencies looks like, so F1 carries meaningful signal when comparing.

    Returns:
        Dictionary mapping model name to (preprocessor, model) Pipeline.
    """
    preprocessor = build_preprocessor()

    models = {
        "LogReg_default": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(C=1.0, random_state=42, max_iter=1000, class_weight="balanced"))
        ]),

        "LogReg_L1": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(C=0.1, penalty="l1", solver="saga", random_state=42, max_iter=1000, class_weight="balanced"))
        ]),
        
        "RidgeClassifier": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=42))
        ]),

        "Dummy_most_frequent": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(strategy="most_frequent"))
        ]),

        "Dummy_stratified": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(strategy="stratified", random_state=42))
        ])
    }

    return models


def evaluate_models(models, X, y, cv=5, random_state=42):
    """Run cross-validation on all models and return results.

    Args:
        models: Dictionary of {name: Pipeline}.
        X: Feature DataFrame.
        y: Target Series.
        cv: Number of folds.
        random_state: Random seed.

    Returns:
        DataFrame with columns: model, accuracy_mean, accuracy_std,
        precision_mean, recall_mean, f1_mean.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = []

    for name, pipeline in models.items():
        scores = cross_validate(pipeline, X, y, cv=skf, scoring=["accuracy", "precision", "recall", "f1"])
        results.append({
            "model": name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "precision_mean": scores["test_precision"].mean(),
            "recall_mean": scores["test_recall"].mean(),
            "f1_mean": scores["test_f1"].mean()
        })

    results_df = pd.DataFrame(results)
    return results_df.sort_values(by="f1_mean", ascending=False).reset_index(drop=True)


def final_evaluation(pipeline, X_train, X_test, y_train, y_test):
    """Train a pipeline on full training data and evaluate on the held-out test set.

    Use this on the best model from Task 4 as a final sanity check — the
    test-set metrics should be close to the CV estimates if the model
    generalizes. If they diverge substantially, the CV estimates were
    optimistic and you should investigate.

    Args:
        pipeline: An unfitted sklearn Pipeline (one entry from define_models).
        X_train, X_test: Feature DataFrames (train and held-out test).
        y_train, y_test: Target Series (train and held-out test).

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }

    return metrics


def recommend_model(results_df):
    """Print a recommendation based on the results.

    Args:
        results_df: DataFrame from evaluate_models.
    """
    print("\n=== Model Comparison Table (CV results) ===")
    print(results_df.to_string(index=False))
    print("\n=== Recommendation ===")
    print("Write your recommendation in the PR description.")


def per_class_analysis(models, X_train, y_train, cv=5, random_state=42):
    """Run out-of-fold predictions and build a classification report for each model.

    Args:
        models: Dictionary of {name: Pipeline}.
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        cv: Number of folds.
        random_state: Random seed.

    Returns:
        Dictionary of {model_name: classification_report_dict}.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    reports = {}

    for name, pipeline in models.items():
        y_pred_oof = cross_val_predict(
            pipeline,
            X_train,
            y_train,
            cv=skf
        )

        report = classification_report(
            y_train,
            y_pred_oof,
            output_dict=True,
            zero_division=0
        )

        reports[name] = report

    return reports


def tier1_summary_table(reports):
    """Convert per-class reports into a comparison table."""
    rows = []

    for model_name, report in reports.items():
        rows.append({
            "model": model_name,
            "class_0_precision": report["0"]["precision"],
            "class_0_recall": report["0"]["recall"],
            "class_0_f1": report["0"]["f1-score"],
            "class_1_precision": report["1"]["precision"],
            "class_1_recall": report["1"]["recall"],
            "class_1_f1": report["1"]["f1-score"]
        })

    return pd.DataFrame(rows).sort_values(by="class_1_f1", ascending=False).reset_index(drop=True)


def print_tier1_reports(reports):
    """Print the per-class metrics in a readable way."""
    print("\n=== Tier 1: Per-Class Analysis ===")

    for model_name, report in reports.items():
        print(f"\n--- {model_name} ---")
        print(
            f"Class 0 (not churned) -> "
            f"Precision: {report['0']['precision']:.3f}, "
            f"Recall: {report['0']['recall']:.3f}, "
            f"F1: {report['0']['f1-score']:.3f}"
        )
        print(
            f"Class 1 (churned)     -> "
            f"Precision: {report['1']['precision']:.3f}, "
            f"Recall: {report['1']['recall']:.3f}, "
            f"F1: {report['1']['f1-score']:.3f}"
        )


def best_minority_class_model(reports):
    """Find the best model for the minority class using class-1 F1."""
    best_model = None
    best_class_1_f1 = -1

    for model_name, report in reports.items():
        class_1_f1 = report["1"]["f1-score"]

        if class_1_f1 > best_class_1_f1:
            best_class_1_f1 = class_1_f1
            best_model = model_name

    return best_model, best_class_1_f1


def compare_logreg_recalls(reports):
    """Compare per-class recall for LogReg_default vs LogReg_L1."""
    default_report = reports["LogReg_default"]
    l1_report = reports["LogReg_L1"]

    print("\n=== Recall Comparison: LogReg_default vs LogReg_L1 ===")
    print(f"Class 0 recall - LogReg_default: {default_report['0']['recall']:.3f}")
    print(f"Class 0 recall - LogReg_L1:      {l1_report['0']['recall']:.3f}")
    print(f"Class 1 recall - LogReg_default: {default_report['1']['recall']:.3f}")
    print(f"Class 1 recall - LogReg_L1:      {l1_report['1']['recall']:.3f}")

if __name__ == "__main__":
    data = load_and_prepare()
    if data is not None:
        X, y = data
        print(f"Data: {X.shape[0]} rows, {X.shape[1]} features")
        print(f"Churn rate: {y.mean():.2%}")

        # Create 80/20 train/test split. The test set is held out for the
        # final evaluation in Task 5 — do not use it during cross-validation.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

        models = define_models()
        if models:
            # Task 4: cross-validation on training data only
            results = evaluate_models(models, X_train, y_train)
            if results is not None:
                recommend_model(results)

                # Select best real model only (exclude dummy models)
                real_models = results[~results["model"].str.contains("Dummy", case=False, na=False)]
                best_model_name = real_models.sort_values(by="f1_mean", ascending=False).iloc[0]["model"]
                best_pipeline = models[best_model_name]

                print(f"\nBest real model based on CV F1: {best_model_name}")

                # Final evaluation on held-out test set
                test_metrics = final_evaluation(best_pipeline, X_train, X_test, y_train, y_test)

                print("\n=== Final Evaluation on Held-Out Test Set ===")
                for metric, value in test_metrics.items():
                    print(f"{metric}: {value:.3f}")

                # Compare with CV F1
                best_cv_row = real_models[real_models["model"] == best_model_name].iloc[0]
                print("\n=== CV vs Test Comparison ===")
                print(f"CV F1 Mean: {best_cv_row['f1_mean']:.3f}")
                print(f"CV Accuracy Mean: {best_cv_row['accuracy_mean']:.3f}")
                print(f"Test F1: {test_metrics['f1']:.3f}")
                print(f"Test Accuracy: {test_metrics['accuracy']:.3f}")

                # Tier 1: Per-class analysis using out-of-fold predictions
                reports = per_class_analysis(models, X_train, y_train)

                print_tier1_reports(reports)

                tier1_df = tier1_summary_table(reports)
                print("\n=== Tier 1 Summary Table ===")
                print(tier1_df.to_string(index=False))

                best_class1_model, best_class1_f1 = best_minority_class_model(reports)
                print(
                    f"\nBest model for minority class (class 1 / churned) "
                    f"based on class-1 F1: {best_class1_model} ({best_class1_f1:.3f})"
                )

                compare_logreg_recalls(reports)


"""
Recommendation:
I recommend RidgeClassifier for this task,
because it achieved the highest F1 score among the real models in cross-validation (0.341)
and also gave a slightly stronger balance between precision and recall. Accuracy alone is not enough here,
because this dataset is imbalanced, and the Dummy_most_frequent model got the highest accuracy (0.838)
while completely failing to detect churners, with 0.000 precision, recall, and F1. 
The RidgeClassifier did much better at identifying churned customers, 
reaching 0.621 recall in cross-validation and 0.653 recall on the held-out test set, 
which is important because missing a customer who may leave is more costly than a false alarm. 
It also clearly outperformed the Dummy_stratified baseline on F1 (0.341 vs 0.173), 
which shows it learned useful patterns beyond random guessing.
Since the final test F1 (0.381) was a bit higher than the CV F1 (0.341), 
the recommendation is supported and the model appears to generalize reasonably well to unseen data.
"""