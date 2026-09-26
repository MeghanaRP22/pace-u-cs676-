"""
train.py — fit the URL model's weights from labelled data.

    uv run python train.py

Writes:
    model_weights.json          read by credibility.py at run time
    results/train_report.json   cross-validated metrics, calibration, chosen penalties
    results/lasso_path.csv      which features survive as the L1 penalty grows
    results/cv_predictions.csv  out-of-fold predictions for every training URL

THE MODEL
    Fractional logit (Papke & Wooldridge, 1996): the label y in [0, 1] is
    modelled as E[y | x] = sigmoid(b + w·x), fitted by minimising binomial
    cross-entropy with fractional targets plus an L2 penalty on w. The link keeps
    every prediction inside (0, 1) and makes stacked penalties saturate instead
    of adding without a floor (weakness 12). Fitted by Newton's method.

WHY THESE CHOICES
    * ~190 examples, ~30 features: some penalty is needed; L2 is chosen by
      10-fold cross-validation.
    * Lasso (L1, proximal gradient) is fitted separately along a penalty path
      purely to REPORT which features carry signal (Session 06), not to score.
    * 200 bootstrap refits give the parameter uncertainty; their out-of-bag
      residuals, split by known/unknown domain, give the noise term.
      Together they form the 90% interval shown in the app (Session 05).

Needs numpy (installed by `uv sync` as a Streamlit dependency). The scorer
itself does not: inference in credibility.py is pure Python.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

import credibility as cred
from training_data import TRAINING_URLS

SEED = 676
N_BOOT = 200
N_FOLDS = 10
LAMBDA_GRID = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
LASSO_GRID = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
RESULTS = os.path.join(cred.HERE, "results")


# -----------------------------------------------------------------------------
# Data: build the design matrix and refuse to train on anything held out.
# The leakage check compares domains, not just URLs, because a held-out domain
# seen in training would make the held-out numbers meaningless.
# -----------------------------------------------------------------------------
def check_no_leakage() -> None:
    import evaluate

    eval_urls = {u for u, _, _ in evaluate.ALL_EVAL_URLS}
    held_domains = {cred._registered_domain(cred._normalize_domain(u))
                    for u, _, _ in evaluate.HELD_OUT_URLS + evaluate.EXTENDED_URLS}
    problems = []
    for url, _, _ in TRAINING_URLS:
        if url in eval_urls:
            problems.append(f"URL also in evaluate.py: {url}")
        dom = cred._registered_domain(cred._normalize_domain(url))
        if dom in held_domains:
            problems.append(f"held-out domain used in training: {dom} ({url})")
    if problems:
        print("Refusing to train — training data leaks into the evaluation set:")
        for p in problems:
            print("  ", p)
        sys.exit(1)


def design(use_ratings: bool) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
    rows, ys, known = [], [], []
    for url, label, _ in TRAINING_URLS:
        f = cred.extract_features(url, use_ratings=use_ratings)
        rows.append([f[n] for n in cred.FEATURE_NAMES])
        ys.append(label)
        known.append(bool(f["has_curated"] or f["has_ratings"]))
    return np.array(rows, float), np.array(ys, float), known


# -----------------------------------------------------------------------------
# Estimators. Both minimise mean fractional cross-entropy; ridge adds
# (lam/2)||w||^2 and uses Newton steps, lasso adds lam*||w||_1 and uses
# proximal gradient (ISTA). The intercept is never penalised.
# -----------------------------------------------------------------------------
def _sig(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float, iters: int = 50) -> Tuple[float, np.ndarray]:
    n, d = X.shape
    A = np.hstack([np.ones((n, 1)), X])
    beta = np.zeros(d + 1)
    beta[0] = np.log(y.mean() / (1 - y.mean()))
    P = np.eye(d + 1) * lam
    P[0, 0] = 0.0
    for _ in range(iters):
        p = _sig(A @ beta)
        grad = A.T @ (p - y) / n + P @ beta
        H = (A.T * (p * (1 - p))) @ A / n + P + 1e-8 * np.eye(d + 1)
        step = np.linalg.solve(H, grad)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return float(beta[0]), beta[1:]


def fit_lasso(X: np.ndarray, y: np.ndarray, lam: float, iters: int = 20000) -> Tuple[float, np.ndarray]:
    n, d = X.shape
    A = np.hstack([np.ones((n, 1)), X])
    L = 0.25 * np.linalg.eigvalsh(A.T @ A / n).max()
    t = 1.0 / L
    beta = np.zeros(d + 1)
    for _ in range(iters):
        p = _sig(A @ beta)
        g = A.T @ (p - y) / n
        nb = beta - t * g
        nb[1:] = np.sign(nb[1:]) * np.maximum(np.abs(nb[1:]) - t * lam, 0.0)
        if np.abs(nb - beta).max() < 1e-9:
            beta = nb
            break
        beta = nb
    return float(beta[0]), beta[1:]


def predict(b0: float, w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return _sig(b0 + X @ w)


def band(v: np.ndarray) -> np.ndarray:
    return np.where(v >= 0.70, 2, np.where(v >= 0.40, 1, 0))


# -----------------------------------------------------------------------------
# Cross-validation: out-of-fold predictions for every training URL, used both
# to pick the penalty and to measure calibration honestly.
# -----------------------------------------------------------------------------
def folds(n: int) -> List[np.ndarray]:
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    return [idx[k::N_FOLDS] for k in range(N_FOLDS)]


def cv_predictions(X, y, fitter, lam) -> np.ndarray:
    out = np.zeros_like(y)
    for test in folds(len(y)):
        train = np.setdiff1d(np.arange(len(y)), test)
        b0, w = fitter(X[train], y[train], lam)
        out[test] = predict(b0, w, X[test])
    return out


def calibration(pred: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """Reliability bins, calibration slope/intercept, and a band-level Brier score."""
    edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    bins = []
    for lo, hi in zip(edges, edges[1:]):
        m = (pred >= lo) & (pred < hi)
        if m.any():
            bins.append({"range": f"{lo:.1f}-{min(hi, 1):.1f}", "n": int(m.sum()),
                         "mean_pred": round(float(pred[m].mean()), 3),
                         "mean_label": round(float(y[m].mean()), 3)})
    slope, intercept = np.polyfit(pred, y, 1)
    # Brier score for the binary event "label is HIGH", treating the score as a probability.
    brier_high = float(np.mean((pred - (y >= 0.70)) ** 2))
    return {"bins": bins, "slope": round(float(slope), 3), "intercept": round(float(intercept), 3),
            "brier_high": round(brier_high, 4)}


def train_variant(name: str, use_ratings: bool) -> Tuple[Dict, Dict]:
    X, y, known = design(use_ratings)
    rng = np.random.default_rng(SEED)

    # Choose the ridge penalty by 10-fold CV MAE.
    cv_scores = {}
    for lam in LAMBDA_GRID:
        p = cv_predictions(X, y, fit_ridge, lam)
        cv_scores[lam] = float(np.abs(p - y).mean())
    lam = min(cv_scores, key=cv_scores.get)
    oof = cv_predictions(X, y, fit_ridge, lam)
    b0, w = fit_ridge(X, y, lam)

    # Bootstrap: parameter draws plus out-of-bag residuals by group.
    boot_b0, boot_w = [], []
    res_known, res_unknown = [], []
    known_arr = np.array(known)
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        bb0, bw = fit_ridge(X[idx], y[idx], lam, iters=30)
        boot_b0.append(round(float(bb0), 4))
        boot_w.append([round(float(v), 4) for v in bw])
        oob = np.setdiff1d(np.arange(len(y)), idx)
        r = y[oob] - predict(bb0, bw, X[oob])
        res_known.extend(r[known_arr[oob]].tolist())
        res_unknown.extend(r[~known_arr[oob]].tolist())
    q = np.linspace(0, 1, 101)
    rk = np.quantile(res_known, q).round(4).tolist() if res_known else [0.0]
    ru = np.quantile(res_unknown, q).round(4).tolist() if res_unknown else [0.0]
    # Shuffle the quantile grid deterministically so pairing with bootstrap
    # draws is not monotone (credibility.py indexes it with a stride).
    rk = list(np.array(rk)[rng.permutation(len(rk))])
    ru = list(np.array(ru)[rng.permutation(len(ru))])

    # Lasso path — reporting only.
    lasso_rows = []
    lasso_cv = {}
    for lam1 in LASSO_GRID:
        lb0, lw = fit_lasso(X, y, lam1)
        p = cv_predictions(X, y, fit_lasso, lam1)
        lasso_cv[lam1] = float(np.abs(p - y).mean())
        lasso_rows.append({"lambda": lam1, "cv_mae": round(lasso_cv[lam1], 4),
                           "nonzero": int((np.abs(lw) > 1e-6).sum()),
                           **{n: round(float(v), 3) for n, v in zip(cred.FEATURE_NAMES, lw)}})

    report = {
        "variant": name, "n_train": int(len(y)), "n_features": int(X.shape[1]),
        "ridge_lambda": lam, "ridge_cv_mae_by_lambda": {str(k): round(v, 4) for k, v in cv_scores.items()},
        "cv_mae": round(float(np.abs(oof - y).mean()), 4),
        "cv_band_accuracy": round(float((band(oof) == band(y)).mean()), 4),
        "cv_worst": round(float(np.abs(oof - y).max()), 4),
        "calibration": calibration(oof, y),
        "coefficients": {n: round(float(v), 3) for n, v in zip(cred.FEATURE_NAMES, w)},
        "intercept": round(b0, 3),
        "coef_boot_sd": {n: round(float(s), 3) for n, s in zip(cred.FEATURE_NAMES, np.std(boot_w, axis=0))},
        "lasso_best_lambda": min(lasso_cv, key=lasso_cv.get),
    }
    weights = {"intercept": round(b0, 5), "coef": [round(float(v), 5) for v in w],
               "boot_intercept": boot_b0, "boot_coef": boot_w,
               "residuals_known": [float(v) for v in rk], "residuals_unknown": [float(v) for v in ru],
               "lambda": lam}
    return weights, {"report": report, "lasso": lasso_rows, "oof": oof.tolist(), "y": y.tolist()}


def main() -> None:
    check_no_leakage()
    os.makedirs(RESULTS, exist_ok=True)
    variants, reports = {}, {}
    for name, use_ratings in (("full", True), ("structural", False)):
        weights, extra = train_variant(name, use_ratings)
        variants[name] = weights
        reports[name] = extra["report"]
        with open(os.path.join(RESULTS, f"lasso_path_{name}.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(extra["lasso"][0].keys()))
            wr.writeheader()
            wr.writerows(extra["lasso"])
        if name == "full":
            with open(os.path.join(RESULTS, "cv_predictions.csv"), "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["url", "category", "label", "cv_prediction"])
                for (url, label, cat), p in zip(TRAINING_URLS, extra["oof"]):
                    wr.writerow([url, cat, label, round(p, 4)])
        r = extra["report"]
        print(f"[{name:10}] n={r['n_train']}  lambda={r['ridge_lambda']}  CV MAE={r['cv_mae']:.3f}  "
              f"CV band acc={r['cv_band_accuracy']:.1%}  calib slope={r['calibration']['slope']}")

    with open(cred.WEIGHTS_PATH, "w", encoding="utf-8") as fh:
        json.dump({"feature_names": cred.FEATURE_NAMES, "seed": SEED, "n_boot": N_BOOT,
                   "variants": variants}, fh, separators=(",", ":"))
    with open(os.path.join(RESULTS, "train_report.json"), "w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2)
    print(f"wrote {os.path.relpath(cred.WEIGHTS_PATH)} and results/")


if __name__ == "__main__":
    main()
