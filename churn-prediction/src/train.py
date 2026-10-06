"""Train and compare churn models on the real Telco dataset, then save the best one.

Usage:  python src/train.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download 'Telco Customer Churn' from Kaggle and save it as "
            "data/Telco-Customer-Churn.csv")
    return pd.read_csv(DATA_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 11 rows have a blank TotalCharges. They are brand-new customers (tenure = 0),
    # so their total charges so far are really 0. We keep them instead of dropping.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df = df.drop(columns=["customerID"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def build_pipeline(model, categorical):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("prep", pre), ("model", model)])


def metrics(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = clean(load_data())
    print(f"Rows: {len(df)} | Churn rate: {df['Churn'].mean():.1%}")

    X, y = df.drop(columns="Churn"), df["Churn"]
    categorical = [c for c in X.columns if c not in NUMERIC]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 1) Compare three models: 5-fold CV on the training set + score on the held-out test set
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    rows, fitted = [], {}
    for name, model in candidates.items():
        pipe = build_pipeline(model, categorical)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        row = {"model": name, "cv_roc_auc": cv_auc.mean(),
               **metrics(y_test, pipe.predict_proba(X_test)[:, 1])}
        rows.append(row)
        fitted[name] = pipe
        print(f"  trained {name}")

    results = pd.DataFrame(rows).set_index("model").round(4)
    print("\n", results, "\n")

    # 2) Tune the best model (chosen by CV ROC-AUC, not by the test set)
    best_name = results["cv_roc_auc"].idxmax()
    grids = {
        "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10]},
        "Random Forest": {"model__max_depth": [6, 10, None],
                          "model__min_samples_leaf": [3, 5, 10]},
        "Gradient Boosting": {"model__n_estimators": [100, 200],
                              "model__learning_rate": [0.03, 0.05, 0.1],
                              "model__max_depth": [2, 3]},
    }
    search = GridSearchCV(build_pipeline(candidates[best_name], categorical),
                          grids[best_name], cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print(f"Best model: {best_name} | tuned params: {search.best_params_}")
    tuned = {"model": f"{best_name} (tuned)", "cv_roc_auc": search.best_score_,
             **metrics(y_test, best.predict_proba(X_test)[:, 1])}
    results.loc[tuned["model"]] = pd.Series(tuned).drop("model").round(4)

    # 3) Pick the decision threshold using only training data (cross-validated predictions)
    oof = cross_val_predict(build_pipeline(search.best_estimator_.named_steps["model"],
                                           categorical), X_train, y_train,
                            cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, oof)
    f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    threshold = float(thr[np.argmax(f1s)])
    final = metrics(y_test, best.predict_proba(X_test)[:, 1], threshold)
    results.loc[f"{best_name} (tuned, threshold {threshold:.2f})"] = pd.Series(
        {"cv_roc_auc": search.best_score_, **final}).round(4)
    print(f"Chosen threshold: {threshold:.2f}\n")
    print(results, "\n")
    results.to_csv(REPORTS_DIR / "model_comparison.csv")

    # Charts ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC curves (test set)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    proba_test = best.predict_proba(X_test)[:, 1]
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, (proba_test >= threshold).astype(int),
        display_labels=["Stay", "Churn"], cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix: {best_name}\n(threshold {threshold:.2f})")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=10,
                                 random_state=RANDOM_STATE, n_jobs=-1)
    imp_s = pd.Series(imp.importances_mean, index=X_test.columns).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(7, 5))
    imp_s.plot.barh(ax=ax, color="#3b7ddd")
    ax.set_title("What drives churn? (permutation importance)")
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    # Churn rate by contract type (simple business insight chart)
    fig, ax = plt.subplots(figsize=(5, 4))
    (df.groupby("Contract")["Churn"].mean() * 100).plot.bar(ax=ax, color="#e07b39")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn rate by contract type")
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "churn_by_contract.png", dpi=150)
    plt.close(fig)

    # Save -------------------------------------------------------------------
    joblib.dump(best, MODELS_DIR / "churn_model.joblib")
    meta = {"best_model": best_name, "threshold": threshold,
            "best_params": {k: (v if v is None else float(v) if isinstance(v, float) else v)
                            for k, v in search.best_params_.items()},
            "metrics": {k: float(v) for k, v in final.items()},
            "features": list(X.columns)}
    (MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved model to {MODELS_DIR / 'churn_model.joblib'}")
    print(f"Saved charts/tables to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
