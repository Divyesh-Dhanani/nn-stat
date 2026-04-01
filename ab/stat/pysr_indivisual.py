"""
PySR symbolic regression for a single dataset and epoch, with optional 5‑fold CV.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from pysr import PySRRegressor

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
STAGE1_TOP_N = 3          # top 3 features
STAGE2_TOP_N = 5          # top 5 features
R2_MIN_GAIN = 0.05        # gain needed to proceed to stage 3
SCORE_THRESHOLD = 0.2     # dcor or mic must exceed this to be "significant"

PYSR_ITERATIONS = 300
PYSR_POPULATIONS = 30
PYSR_MAXSIZE = 30
PYSR_SEED = 42

BINARY_OPS = ["+", "-", "*", "/", "^"]
UNARY_OPS = ["sqrt", "log", "abs", "exp", "square"]

TARGET_COL = "accuracy"

# Paths (relative to script location)
_base = os.path.join(os.path.dirname(__file__), "..", "..")
DCOR_OUT_DIR = os.path.join(_base, "dcor_out")
MIC_OUT_DIR = os.path.join(_base, "mic_out")
DATASET_DIR = os.path.join(_base, "dataset_splits", "epoch_1_5_50")
PYSR_OUT_DIR = os.path.join(_base, "pysr_out")

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_best_r2_and_index(model, X: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    """Return (best_r2, best_index) of equation with max R²."""
    best_r2 = -np.inf
    best_idx = None
    for idx in model.equations_["complexity"]:
        try:
            y_pred = model.predict(X, index=int(idx))
            r2 = r2_score(y, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_idx = idx
        except Exception:
            continue
    return best_r2, best_idx


def run_pysr_stage(X: np.ndarray, y: np.ndarray,
                   feature_names: List[str], stage_label: str,
                   outdir: str) -> Tuple[pd.DataFrame, PySRRegressor]:
    """Run PySR and return equations DataFrame and fitted model."""
    print(f"\n  {stage_label}")
    print(f"  Features: {feature_names}")
    print(f"  n = {len(y):,}  |  iterations = {PYSR_ITERATIONS}")

    model = PySRRegressor(
        niterations=PYSR_ITERATIONS,
        populations=PYSR_POPULATIONS,
        maxsize=PYSR_MAXSIZE,
        binary_operators=BINARY_OPS,
        unary_operators=UNARY_OPS,
        model_selection="best",
        random_state=PYSR_SEED,
        verbosity=1,
        temp_equation_file=True,
        delete_tempfiles=True,
    )

    t0 = time.time()
    model.fit(X, y, variable_names=feature_names)
    elapsed = time.time() - t0

    equations = model.equations_
    print(f"  Done in {elapsed:.1f}s | {len(equations)} Pareto‑optimal equations")

    # Save equations CSV
    csv_path = os.path.join(outdir, f"{stage_label.replace(' ', '_').lower()}_equations.csv")
    equations.to_csv(csv_path, index=False)

    return equations, model


def plot_pareto(stage_results: List[Tuple[str, pd.DataFrame]], outpath: str) -> None:
    """Plot Pareto front (complexity vs loss) for all stages."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (label, equations), color in zip(stage_results, colors):
        if equations is None or equations.empty:
            continue
        ax.plot(equations["complexity"], equations["loss"],
                "o-", color=color, label=label, linewidth=1.5, markersize=5)
    ax.set_xlabel("Formula Complexity (nodes)")
    ax.set_ylabel("MSE Loss")
    ax.set_title("PySR Pareto Front — Complexity vs Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_r2_bars(stage_r2: dict, outpath: str) -> None:
    """Bar chart of best R² per stage."""
    labels = list(stage_r2.keys())
    values = [stage_r2[k] for k in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values,
                  color=["#1f77b4", "#ff7f0e", "#2ca02c"][:len(labels)],
                  edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Best R²")
    ax.set_title("PySR — Best R² per Stage")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--epoch", required=True, type=int, help="Epoch number")
    parser.add_argument("--cv", action="store_true", help="Perform 5‑fold CV on the best equation")
    args = parser.parse_args()

    dataset = args.dataset
    epoch = args.epoch

    # Define paths
    dcor_csv = os.path.join(DCOR_OUT_DIR, dataset, f"epoch_{epoch}.csv")
    mic_csv = os.path.join(MIC_OUT_DIR, dataset, f"epoch_{epoch}.csv")
    data_csv = os.path.join(DATASET_DIR, f"{dataset}.csv")
    outdir = os.path.join(PYSR_OUT_DIR, dataset, f"epoch_{epoch}")
    ensure_dir(outdir)

    # Check inputs
    for fpath in [dcor_csv, mic_csv, data_csv]:
        if not os.path.exists(fpath):
            print(f"ERROR: missing {fpath}")
            return

    # 1. Load dcor and mic results, combine
    dcor_df = pd.read_csv(dcor_csv)[["feature", "dcor", "spearman"]]
    mic_df = pd.read_csv(mic_csv)[["feature", "mic"]]
    scores = dcor_df.merge(mic_df, on="feature", how="outer").fillna(0)

    # 2. Select significant features
    significant = scores[(scores["dcor"] > SCORE_THRESHOLD) |
                         (scores["mic"] > SCORE_THRESHOLD)].copy()
    significant.sort_values("dcor", ascending=False, inplace=True)
    sig_features = significant["feature"].tolist()

    print(f"\nSignificant features ({len(sig_features)}):")
    for i, row in significant.iterrows():
        print(f"  {i+1:2d}. {row['feature']:32s}  dcor={row['dcor']:.4f}  "
              f"mic={row['mic']:.4f}  spearman={row['spearman']:+.4f}")

    if not sig_features:
        print("No significant features. Exiting.")
        return

    # 3. Load data, filter epoch, compute z-score of accuracy
    df = pd.read_csv(data_csv, low_memory=False)
    df = df[df["epoch"] == epoch].copy()
    if df.empty:
        print(f"No data for epoch {epoch}")
        return

    df[TARGET_COL] = df["accuracy"]

    # Keep only needed columns and drop NaNs
    keep_cols = [TARGET_COL] + [f for f in sig_features if f in df.columns]
    df = df[keep_cols].dropna(subset=[TARGET_COL])

    # 4. Define feature sets for stages
    top3 = sig_features[:STAGE1_TOP_N]
    top5 = sig_features[:STAGE2_TOP_N]
    stage_sets = {
        f"Stage1_top{STAGE1_TOP_N}": top3,
        f"Stage2_top{STAGE2_TOP_N}": top5,
        f"Stage3_all{len(sig_features)}": sig_features,
    }

    # 5. Run stages
    stage_results = []          # (label, equations_df)
    stage_r2 = {}
    stage_models = {}
    stage_data = {}             # label -> (X, y, features)
    stage_best_idx = {}         # label -> best equation index
    run_stage3 = False

    for stage_idx, (label, feats) in enumerate(stage_sets.items(), 1):
        # Stage 3 only if gain >= threshold
        if stage_idx == 3 and not run_stage3:
            print(f"\nSkipping {label} (R² gain from stage 1→2 < {R2_MIN_GAIN})")
            stage_results.append((label, None))
            continue

        # Prepare data for this stage
        X = df[feats].values.astype(float)
        y = df[TARGET_COL].values.astype(float)

        # Drop rows with NaN in any selected feature
        mask = ~np.isnan(X).any(axis=1)
        X, y = X[mask], y[mask]
        if len(y) < 10:
            print(f"Not enough rows after dropping NaNs ({len(y)}), skipping {label}")
            stage_results.append((label, None))
            continue

        # Run PySR
        equations, model = run_pysr_stage(X, y, feats, label, outdir)
        r2, best_idx = compute_best_r2_and_index(model, X, y)

        stage_results.append((label, equations))
        stage_r2[label] = r2
        stage_models[label] = model
        stage_data[label] = (X, y, feats)
        stage_best_idx[label] = best_idx

        # Determine if we should run stage 3
        if stage_idx == 2:
            r2_s1 = stage_r2.get(list(stage_sets.keys())[0], -np.inf)
            gain = r2 - r2_s1
            print(f"  R² gain Stage1→Stage2: {gain:+.4f} (threshold {R2_MIN_GAIN})")
            run_stage3 = gain >= R2_MIN_GAIN
            if not run_stage3:
                print("  → Stage 3 will be skipped")
            else:
                print("  → Stage 3 will run")

    # 6. Determine the best stage
    if not stage_r2:
        print("No stages completed. Exiting.")
        return

    best_stage_label = max(stage_r2, key=stage_r2.get)
    best_stage_r2 = stage_r2[best_stage_label]
    best_X, best_y, best_feats = stage_data[best_stage_label]
    best_model = stage_models[best_stage_label]
    best_eq_idx = stage_best_idx[best_stage_label]

    # Save best equation string for reference
    best_eq_str = best_model.equations_.loc[best_model.equations_["complexity"] == best_eq_idx, "sympy_format"].values[0]

    print(f"\nBest stage: {best_stage_label} (R² = {best_stage_r2:.4f})")
    print(f"Best equation: {best_eq_str}")

    # 7. Cross‑validation (if requested)
    if args.cv:
        print(f"\nPerforming 5‑fold cross‑validation on best stage: {best_stage_label}")
        print(f"Features: {best_feats}")
        print(f"n = {len(best_y)}")

        kf = KFold(n_splits=5, shuffle=True, random_state=PYSR_SEED)
        cv_algo_r2 = []
        cv_ref_r2 = []
        fold = 1

        cv_dir = os.path.join(outdir, "cv")
        ensure_dir(cv_dir)

        for train_idx, val_idx in kf.split(best_X):
            X_train, X_val = best_X[train_idx], best_X[val_idx]
            y_train, y_val = best_y[train_idx], best_y[val_idx]

            # Train a new model on the training fold
            model_fold = PySRRegressor(
                niterations=PYSR_ITERATIONS,
                populations=PYSR_POPULATIONS,
                maxsize=PYSR_MAXSIZE,
                binary_operators=BINARY_OPS,
                unary_operators=UNARY_OPS,
                model_selection="best",
                random_state=PYSR_SEED,
                verbosity=0,                 # reduce output
                temp_equation_file=True,
                delete_tempfiles=True,
            )
            model_fold.fit(X_train, y_train, variable_names=best_feats)

            # Best equation on the training set
            _, best_idx_fold = compute_best_r2_and_index(model_fold, X_train, y_train)
            y_pred_algo = model_fold.predict(X_val, index=int(best_idx_fold))
            r2_val_algo = r2_score(y_val, y_pred_algo)

            # Reference equation on the validation set
            y_pred_ref = best_model.predict(X_val, index=int(best_eq_idx))
            r2_val_ref = r2_score(y_val, y_pred_ref)

            cv_algo_r2.append(r2_val_algo)
            cv_ref_r2.append(r2_val_ref)
            print(f"  Fold {fold}: algo R²_val = {r2_val_algo:.4f}, ref R²_val = {r2_val_ref:.4f}")
            fold += 1

        # Summary statistics
        mean_algo = np.mean(cv_algo_r2)
        std_algo = np.std(cv_algo_r2)
        mean_ref = np.mean(cv_ref_r2)
        std_ref = np.std(cv_ref_r2)

        with open(os.path.join(outdir, "cv_results.txt"), "w") as f:
            f.write(f"Cross‑validation results for best stage \"{best_stage_label}\":\n")
            f.write(f"  Algorithm (best per fold) : R² = {mean_algo:.4f} ± {std_algo:.4f}\n")
            f.write(f"  Reference equation        : R² = {mean_ref:.4f} ± {std_ref:.4f}\n\n")
            f.write("Per‑fold values:\n")
            f.write(f"  Algo: {', '.join(f'{x:.4f}' for x in cv_algo_r2)}\n")
            f.write(f"  Ref : {', '.join(f'{x:.4f}' for x in cv_ref_r2)}\n")

        print(f"\nCV results saved to {os.path.join(outdir, 'cv_results.txt')}")

        # Optional boxplot
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.boxplot([cv_algo_r2, cv_ref_r2], labels=["Algorithm (best per fold)", "Reference equation"])
            ax.set_ylabel("R² on validation set")
            ax.set_title(f"5‑fold CV – {dataset} epoch {epoch}\nBest stage: {best_stage_label}")
            plt.tight_layout()
            fig.savefig(os.path.join(outdir, "cv_boxplot.png"), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"Could not plot boxplot: {e}")

    # 8. Generate plots and summary (original)
    valid = [(lbl, eq) for lbl, eq in stage_results if eq is not None]
    if valid:
        plot_pareto(valid, os.path.join(outdir, "pareto_front.png"))
    if stage_r2:
        plot_r2_bars(stage_r2, os.path.join(outdir, "r2_comparison.png"))

    # 9. Save text summary
    with open(os.path.join(outdir, "best_equations.txt"), "w") as f:
        f.write(f"PySR Symbolic Regression — {dataset} epoch {epoch}\n")
        f.write("=" * 70 + "\n\n")
        for stage_idx, (label, equations) in enumerate(stage_results, 1):
            if equations is None:
                continue
            f.write(f"{label}\n")
            f.write(f"Features: {list(stage_sets.values())[stage_idx-1]}\n")
            f.write(f"Best R²: {stage_r2.get(label, float('nan')):.4f}\n\n")
            f.write("Complexity   Loss        R²      Equation\n")
            f.write("-" * 60 + "\n")
            for _, row in equations.iterrows():
                try:
                    X_cur, y_cur, _ = stage_data[label]
                    y_pred = stage_models[label].predict(X_cur, index=int(row["complexity"]))
                    r2 = r2_score(y_cur, y_pred)
                except:
                    r2 = np.nan
                f.write(f"{int(row['complexity']):>10d}  {row['loss']:>12.6f}  {r2:>8.4f}  {row['sympy_format']}\n")
            f.write("\n")

        if stage_r2:
            best_stage = max(stage_r2, key=stage_r2.get)
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"RECOMMENDED: {best_stage}  (R²={stage_r2[best_stage]:.4f})\n")

    print(f"\nOutput saved to {outdir}")


if __name__ == "__main__":
    main()