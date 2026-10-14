"""Startup growth classifier - demo on SYNTHETIC data (the real dataset is not published).

Pipeline: synthetic startups -> train/test split -> SMOTE on TRAIN only ->
Random Forest + XGBoost soft-voting ensemble -> metrics on the untouched test set -> SHAP.
"""
import numpy as np, pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def make_data(n=450, seed=3):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({
        "revenue_growth_12m": r.normal(0.2, 0.5, n),
        "technical_cofounder": r.integers(0, 2, n),
        "log_funding": r.normal(11, 2, n),
        "b2b": r.integers(0, 2, n),
        "partnerships": r.poisson(2, n),
        "team_size": r.integers(2, 30, n),
    })
    z = (1.6 * d.revenue_growth_12m + 0.8 * d.technical_cofounder + 0.25 * (d.log_funding - 11)
         + 0.4 * d.b2b + 0.15 * d.partnerships - 1.8 + r.normal(0, 0.8, n))
    d["high_growth"] = (z > np.quantile(z, 0.75)).astype(int)  # ~25% minority class
    return d


def main():
    d = make_data()
    X, y = d.drop(columns="high_growth"), d["high_growth"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    Xs, ys = SMOTE(random_state=0).fit_resample(Xtr, ytr)  # resample training data only
    model = VotingClassifier([
        ("rf", RandomForestClassifier(300, random_state=0)),
        ("xgb", XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, eval_metric="logloss")),
    ], voting="soft").fit(Xs, ys)
    p = model.predict_proba(Xte)[:, 1]; pred = (p > 0.5).astype(int)
    print(f"accuracy {accuracy_score(yte, pred):.2f} | precision {precision_score(yte, pred):.2f} | "
          f"recall {recall_score(yte, pred):.2f} | AUC {roc_auc_score(yte, p):.2f}  (synthetic data)")
    import shap
    xgb = model.named_estimators_["xgb"]
    sv = shap.TreeExplainer(xgb).shap_values(Xte)
    imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
    print("\nmean |SHAP| (XGBoost):\n", imp.round(3).to_string())


if __name__ == "__main__":
    main()
