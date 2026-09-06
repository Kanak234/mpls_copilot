"""
train.py
--------
Trains the predictive fault model and reports the metrics that matter for
the challenge: precision, recall, false-positive rate, and - most importantly -
PREDICTION LEAD TIME (how many steps before impact we raise the alarm).

MODEL CHOICE:
We use HistGradientBoostingClassifier (sklearn). Why not a giant LSTM?
  - It trains in seconds, runs fully offline, tiny footprint (air-gap friendly).
  - On tabular trend-features it is extremely strong and hard to beat.
  - It is interpretable - we can pull feature importances for the copilot's
    "why" explanation (Copilot Effectiveness score).
The architecture supports swapping in an LSTM later; the interface is the same.

HONEST EVALUATION:
We split by TIME, not randomly. The model trains on early time-steps and is
tested on later ones it has never seen - exactly like real deployment where
you predict the future from the past. A random split would leak future info
and inflate the score dishonestly.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from netcopilot.model.features import build_features, feature_columns


def time_split(feats: pd.DataFrame, train_frac: float = 0.7):
    """Split by step: early steps -> train, later steps -> test."""
    cutoff = feats["step"].quantile(train_frac)
    train = feats[feats["step"] <= cutoff]
    test = feats[feats["step"] > cutoff]
    return train, test, cutoff


def measure_lead_time(test: pd.DataFrame, preds: np.ndarray) -> dict:
    """
    For each link that actually hit impact in the test window, find how many
    steps BEFORE the first impact we first raised an alarm. That gap is the
    lead time the NOC gets to act.
    """
    test = test.copy()
    test["pred"] = preds
    lead_times = []

    for link, g in test.groupby("link"):
        g = g.sort_values("step")
        impact_steps = g[g["fault_phase"] == "impact"]["step"]
        if impact_steps.empty:
            continue
        first_impact = impact_steps.min()
        alarms = g[(g["pred"] == 1) & (g["step"] < first_impact)]["step"]
        if not alarms.empty:
            lead = first_impact - alarms.min()
            lead_times.append(lead)
        else:
            lead_times.append(0)

    return {
        "links_with_impact": len(lead_times),
        "links_caught_early": sum(1 for l in lead_times if l > 0),
        "avg_lead_steps": float(np.mean(lead_times)) if lead_times else 0.0,
        "max_lead_steps": int(np.max(lead_times)) if lead_times else 0,
    }


def _shuffle_importance(model, X, y, n_repeats: int = 3) -> list[float]:
    """
    Lightweight feature importance: shuffle each column, see how much
    F1 drops. Cheap, offline, gives the copilot its 'why' signals.
    We keep X as a DataFrame so column names are preserved (no sklearn warning).
    """
    base = f1_score(y, model.predict(X), zero_division=0)
    rng = np.random.default_rng(0)
    importances = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            f1p = f1_score(y, model.predict(Xp), zero_division=0)
            drops.append(base - f1p)
        importances.append(float(np.mean(drops)))
    return importances


def train_and_eval(csv_path: str = "data/telemetry.csv",
                   model_out: str = "data/predictor.joblib") -> dict:
    raw = pd.read_csv(csv_path)
    feats = build_features(raw)
    cols = feature_columns(feats)

    train, test, cutoff = time_split(feats)
    X_train, y_train = train[cols], train["will_impact"]
    X_test, y_test = test[cols], test["will_impact"]

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=6,
        l2_regularization=1.0, random_state=0,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    lead = measure_lead_time(test, preds)

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": cols}, model_out)

    importances = _shuffle_importance(model, X_test, y_test)
    top = dict(sorted(zip(cols, importances), key=lambda kv: kv[1], reverse=True)[:6])

    return {
        "train_rows": len(train), "test_rows": len(test),
        "split_step": float(cutoff),
        "precision": prec, "recall": rec, "f1": f1,
        "false_positive_rate": fpr,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "lead_time": lead,
        "model_path": model_out,
        "feature_importances": top,
    }


if __name__ == "__main__":
    r = train_and_eval()
    print("=" * 52)
    print(" PREDICTIVE FAULT MODEL - TRAINING REPORT")
    print("=" * 52)
    print(f" train rows / test rows : {r['train_rows']} / {r['test_rows']}")
    print(f" time split at step     : {r['split_step']:.0f}")
    print("-" * 52)
    print(f" Precision              : {r['precision']:.3f}")
    print(f" Recall                 : {r['recall']:.3f}")
    print(f" F1 score               : {r['f1']:.3f}")
    print(f" False-positive rate    : {r['false_positive_rate']:.3f}")
    print(f" Confusion  TP/FP/FN/TN : {r['tp']}/{r['fp']}/{r['fn']}/{r['tn']}")
    print("-" * 52)
    lt = r["lead_time"]
    print(f" Links that hit impact  : {lt['links_with_impact']}")
    print(f" Caught early           : {lt['links_caught_early']}")
    print(f" Avg lead time (steps)  : {lt['avg_lead_steps']:.1f}")
    print(f" Max lead time (steps)  : {lt['max_lead_steps']}")
    print("-" * 52)
    print(" Top signals driving predictions:")
    for feat, imp in r["feature_importances"].items():
        print(f"    {feat:<26} {imp:+.3f}")
    print("=" * 52)
    print(f" model saved -> {r['model_path']}")
