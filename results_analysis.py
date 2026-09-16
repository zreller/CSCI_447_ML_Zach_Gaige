import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from knn_full import (
    min_max_normalize_train_test, load_vote, compute_vdm_tables,
    classification_error, mean_squared_error,
    classification_null_model, regression_null_model,
    pairwise_numeric, pairwise_fires,
    knn_classify, knn_regress,
    edited_nn_classification, edited_nn_regression,
    k_fold_split, tune_knn_regression_numeric, tune_knn_regression_fires,
    tune_k_numeric_classification,
    five_by_two_split, dietterich_5x2cv_ttest,
    RANDOM_STATE,
)

np.random.seed(RANDOM_STATE)
OUT_DIR = os.path.dirname(os.path.abspath(__file__)) or "."
EPSILON_FRACTION = 0.25  

C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_MAGENTA, C_GREEN, C_VIOLET, C_RED = "#e87ba4", "#008300", "#4a3aa7", "#e34948"
C_GRAY = "#898781"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
GAMMA_COLORS = [C_AQUA, C_YELLOW, C_VIOLET, C_RED]  


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=11, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def pairwise_categorical(A_cat, B_cat, vdm_tables, classes, p=1):
    A_cat = np.asarray(A_cat, dtype=object)
    B_cat = np.asarray(B_cat, dtype=object)
    classes = list(classes)
    fallback = np.full(len(classes), 1.0 / len(classes))
    total = np.zeros((len(A_cat), len(B_cat)))
    for j in range(A_cat.shape[1]):
        table = vdm_tables[j]

        def probs_for(vals):
            out = np.empty((len(vals), len(classes)))
            for i, v in enumerate(vals):
                if v in table:
                    Ci, Ci_a = table[v]["Ci"], table[v]["Ci_a"]
                    out[i] = [Ci_a[c] / Ci for c in classes]
                else:
                    out[i] = fallback
            return out

        Pa, Pb = probs_for(A_cat[:, j]), probs_for(B_cat[:, j])
        total += np.sum(np.abs(Pa[:, None, :] - Pb[None, :, :]) ** p, axis=2)
    return total ** (1.0 / p)


def tune_k_categorical_classification(train_df, cat_cols, class_col, k_grid, p=1,
                                       inner_k=5, random_state=None):
    """Same job as knn_full.py's tuner of the same name, but built on the
    fast pairwise_categorical above instead of the slow one."""
    n = len(train_df)
    results = []
    for k in k_grid:
        fold_errors = []
        for tr_idx, val_idx in k_fold_split(n, inner_k, random_state=random_state):
            tr_df, val_df = train_df.iloc[tr_idx], train_df.iloc[val_idx]
            vdm_tables = [compute_vdm_tables(tr_df, col, class_col) for col in cat_cols]
            classes = tr_df[class_col].unique()
            D = pairwise_categorical(val_df[cat_cols].values.tolist(),
                                      tr_df[cat_cols].values.tolist(), vdm_tables, classes, p=p)
            preds = knn_classify(D, tr_df[class_col].values, k)
            fold_errors.append(classification_error(preds, val_df[class_col].values))
        results.append((k, float(np.mean(fold_errors))))
    results.sort(key=lambda t: t[1])
    return results[0][0], results[0][1], results

# Per-dataset-shape analysis runners

def run_numeric_classification(name, df, feature_cols, class_col, k_grid, inner_k=5):
    n = len(df)
    split_idx = int(n * 0.8)
    best_k, _, tuning_results = tune_k_numeric_classification(
        df.iloc[:split_idx].copy(), feature_cols, class_col, k_grid,
        p=2, inner_k=inner_k, random_state=RANDOM_STATE)

    splits = five_by_two_split(n, random_state=RANDOM_STATE)
    null_s, knn_s, ed_s = np.zeros((5, 2)), np.zeros((5, 2)), np.zeros((5, 2))
    kept_fracs, cond_fracs = []
    for rep, (fold_a, fold_b) in enumerate(splits):
        for fi, (tr_idx, te_idx) in enumerate([fold_a, fold_b]):
            tr_df, te_df = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
            tr_df, te_df = min_max_normalize_train_test(tr_df, te_df, feature_cols)

            null_pred = classification_null_model(tr_df[class_col])
            null_s[rep, fi] = classification_error([null_pred] * len(te_df), te_df[class_col].values)

            D_test = pairwise_numeric(te_df[feature_cols].values, tr_df[feature_cols].values, p=2)
            knn_s[rep, fi] = classification_error(
                knn_classify(D_test, tr_df[class_col].values, best_k), te_df[class_col].values)

            D_tt = pairwise_numeric(tr_df[feature_cols].values, tr_df[feature_cols].values, p=2)
            keep = edited_nn_classification(D_tt, tr_df[class_col].values, random_state=0)
            kept_fracs.append(keep.mean())
            D_test_ed = pairwise_numeric(te_df[feature_cols].values, tr_df[feature_cols].values[keep], p=2)
            ed_s[rep, fi] = classification_error(

            cond_keep = condensed_nn_classification(D_tt, tr_y, random_state=0)
            cond_fracs.append(cond_keep.mean())
            cond_s[rep, fi] = classification_error(
                knn_classify(D_test[:, cond_keep], tr_y[cond_keep], best_k), te_y)
            
    return _package_result(name, "classification", {"k": best_k}, tuning_results,
                            null_s, knn_s, ed_s, kept_fracs, ed
                          -s, cond_s, ed_fracs. cond_fracs)


def run_categorical_classification(name, df, cat_cols, class_col, k_grid, inner_k=5):
    n = len(df)
    split_idx = int(n * 0.8)
    best_k, _, tuning_results = tune_k_categorical_classification(
        df.iloc[:split_idx], cat_cols, class_col, k_grid, p=1, inner_k=inner_k, random_state=RANDOM_STATE)

    splits = five_by_two_split(n, random_state=RANDOM_STATE)
    null_s, knn_s, ed_s, cond_s = np.zeros((5, 2)), np.zeros((5, 2)), np.zeros((5, 2))
    kept_fracs, cond_fracs = []
    for rep, (fold_a, fold_b) in enumerate(splits):
        for fi, (tr_idx, te_idx) in enumerate([fold_a, fold_b]):
            tr_df, te_df = df.iloc[tr_idx], df.iloc[te_idx]
            vdm_tables = [compute_vdm_tables(tr_df, col, class_col) for col in cat_cols]
            classes = tr_df[class_col].unique()
            X_tr, X_te = tr_df[cat_cols].values.tolist(), te_df[cat_cols].values.tolist()

            null_pred = classification_null_model(tr_df[class_col])
            null_s[rep, fi] = classification_error([null_pred] * len(te_df), te_df[class_col].values)

            D_test = pairwise_categorical(X_te, X_tr, vdm_tables, classes, p=1)
            knn_s[rep, fi] = classification_error(
                knn_classify(D_test, tr_df[class_col].values, best_k), te_df[class_col].values)

            D_tt = pairwise_categorical(X_tr, X_tr, vdm_tables, classes, p=1)
            keep = edited_nn_classification(D_tt, tr_df[class_col].values, random_state=0)
            kept_fracs.append(keep.mean())
            X_tr_keep = [x for x, m in zip(X_tr, keep) if m]
            D_test_ed = pairwise_categorical(X_te, X_tr_keep, vdm_tables, classes, p=1)
            ed_s[rep, fi] = classification_error(

            cond_keep = condensed_nn_classification(D_tt, tr_y, random_state=0)
            cond_fracs.append(cond_keep.mean())
            cond_s[rep, fi] = classification_error(
                knn_classify(D_test[:, cond_keep], tr_y[cond_keep], best_k), te_y)

    return _package_result(name, "classification", {"k": best_k}, tuning_results,
                            null_s, knn_s, ed_s, kept_fracs, cond_s,ed_fracs, cond_fracs)


def run_numeric_regression(name, df, feature_cols, target_col, k_grid, gamma_grid, inner_k=5):
    n = len(df)
    split_idx = int(n * 0.8)
    best_params, _, tuning_results = tune_knn_regression_numeric(
        df.iloc[:split_idx].copy(), feature_cols, target_col, k_grid, gamma_grid,
        p=2, inner_k=inner_k, random_state=RANDOM_STATE)
    best_k, best_gamma = best_params["k"], best_params["gamma"]

    splits = five_by_two_split(n, random_state=RANDOM_STATE)
    null_s, knn_s, ed_s, cond_s = np.zeros((5, 2)), np.zeros((5, 2)), np.zeros((5, 2))
    kept_fracs, cond_fracs = []
    for rep, (fold_a, fold_b) in enumerate(splits):
        for fi, (tr_idx, te_idx) in enumerate([fold_a, fold_b]):
            tr_df, te_df = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
            tr_df, te_df = min_max_normalize_train_test(tr_df, te_df, feature_cols)

            null_pred = regression_null_model(tr_df[target_col])
            null_s[rep, fi] = mean_squared_error([null_pred] * len(te_df), te_df[target_col].values)

            D_test = pairwise_numeric(te_df[feature_cols].values, tr_df[feature_cols].values, p=2)
            knn_s[rep, fi] = mean_squared_error(
                knn_regress(D_test, tr_df[target_col].values, best_k, best_gamma), te_df[target_col].values)

            epsilon = EPSILON_FRACTION * np.std(tr_df[target_col].values)
            D_tt = pairwise_numeric(tr_df[feature_cols].values, tr_df[feature_cols].values, p=2)
            keep = edited_nn_regression(D_tt, tr_df[target_col].values, epsilon=epsilon)
            kept_fracs.append(keep.mean())
            D_test_ed = pairwise_numeric(te_df[feature_cols].values, tr_df[feature_cols].values[keep], p=2)
            ed_s[rep, fi] = mean_squared_error(

            
            cond_keep = condensed_nn_regression(D_tt, tr_y, epsilon=epsilon, random_state=0)
            cond_fracs.append(cond_keep.mean())
            cond_s[rep, fi] = mean_squared_error(
                knn_regress(D_test_ed, tr_df[target_col].values[keep], best_k, best_gamma), te_df[target_col].values)

    return _package_result(name, "regression", best_params, tuning_results, null_s, knn_s, ed_s, kept_fracs, cond_s, ed_fracs, cond_fracs)


def run_fires_regression(name, df, cols, cyclical_cols, cycle_lengths, norm_cols, target_col,
                          k_grid, gamma_grid, inner_k=5):
    n = len(df)
    split_idx = int(n * 0.8)
    best_params, _, tuning_results = tune_knn_regression_fires(
        df.iloc[:split_idx].copy(), cols, cyclical_cols, cycle_lengths, norm_cols, target_col,
        k_grid, gamma_grid, p=2, inner_k=inner_k, random_state=RANDOM_STATE)
    best_k, best_gamma = best_params["k"], best_params["gamma"]

    splits = five_by_two_split(n, random_state=RANDOM_STATE)
    null_s, knn_s, ed_s, cond_s = np.zeros((5, 2)), np.zeros((5, 2)), np.zeros((5, 2))
    kept_fracs, cond_fracs = []
    for rep, (fold_a, fold_b) in enumerate(splits):
        for fi, (tr_idx, te_idx) in enumerate([fold_a, fold_b]):
            tr_df, te_df = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
            tr_df, te_df = min_max_normalize_train_test(tr_df, te_df, norm_cols)
            tr_rows, te_rows = tr_df[cols].to_dict("records"), te_df[cols].to_dict("records")

            null_pred = regression_null_model(tr_df[target_col])
            null_s[rep, fi] = mean_squared_error([null_pred] * len(te_df), te_df[target_col].values)

            D_test = pairwise_fires(te_rows, tr_rows, cols, cyclical_cols, cycle_lengths, p=2)
            knn_s[rep, fi] = mean_squared_error(
                knn_regress(D_test, tr_df[target_col].values, best_k, best_gamma), te_df[target_col].values)

            epsilon = EPSILON_FRACTION * np.std(tr_df[target_col].values)
            D_tt = pairwise_fires(tr_rows, tr_rows, cols, cyclical_cols, cycle_lengths, p=2)
            keep = edited_nn_regression(D_tt, tr_df[target_col].values, epsilon=epsilon)
            kept_fracs.append(keep.mean())
            tr_rows_keep = [r for r, m in zip(tr_rows, keep) if m]
            D_test_ed = pairwise_fires(te_rows, tr_rows_keep, cols, cyclical_cols, cycle_lengths, p=2)
            ed_s[rep, fi] = mean_squared_error(
                knn_regress(D_test_ed, tr_df[target_col].values[keep], best_k, best_gamma), te_df[target_col].values)

            cond_keep = condensed_nn_regression(D_tt, tr_y, epsilon=epsilon, random_state=0)
            cond_fracs.append(cond_keep.mean())
            cond_s[rep, fi] = mean_squared_error(
                knn_regress(D_test[:, cond_keep], tr_y[cond_keep], best_k, best_gamma), te_y)


    return _package_result(name, "regression", best_params, tuning_results, null_s, knn_s, ed_s, kept_fracs, cond_s, ed_fracs, cond_fracs)


def _package_result(name, task, best_params, tuning_results, null_s, knn_s, ed_s, kept_fracs):
    t1, p_null_knn, _ = dietterich_5x2cv_ttest(null_s, knn_s)
    t2, p_knn_ed, _ = dietterich_5x2cv_ttest(knn_s, ed_s)
    return {
        "name": name, "task": task, "best_params": best_params, "tuning_results": tuning_results,
        "null_mean": float(null_s.mean()), "knn_mean": float(knn_s.mean()), "edited_mean": float(ed_s.mean()),
        "kept_fraction": float(np.mean(kept_fracs)),
        "p_null_vs_knn": p_null_knn, "p_knn_vs_edited": p_knn_ed,
    }

# Load the six data sets and run the analysis

K_GRID = [1, 3, 5, 7, 9, 11, 15]
GAMMA_GRID = [0.5, 1.0, 2.0, 5.0]

print("Running Breast Cancer...")
cancer_cols = ["id", "clump_thickness", "cell_size_uniformity", "cell_shape_uniformity",
               "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei",
               "bland_chromatin", "normal_nucleoli", "mitoses", "class"]
cancer = pd.read_csv("breast-cancer.data", header=None, names=cancer_cols)
cancer = cancer.drop(columns=["id"])
cancer = cancer[cancer["bare_nuclei"] != "?"]
cancer["bare_nuclei"] = cancer["bare_nuclei"].astype(int)
cancer_cols_norm = ["clump_thickness", "cell_size_uniformity", "cell_shape_uniformity",
                     "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei",
                     "bland_chromatin", "normal_nucleoli", "mitoses"]
cancer = cancer.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
res_cancer = run_numeric_classification("Breast Cancer", cancer, cancer_cols_norm, "class", K_GRID)

print("Running Car Evaluation...")
car_cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
car_cat_cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
car = pd.read_csv("car.data", header=None, names=car_cols)
car = car.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
res_car = run_categorical_classification("Car Evaluation", car, car_cat_cols, "class", K_GRID)

print("Running Congressional Vote...")
vote_cols = ["party", "handicapped-infants", "water-project-cost-sharing",
             "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid",
             "religious-groups-in-schools", "anti-satellite-test-ban",
             "aid-to-nicaraguan-contras", "mx-missile", "immigration",
             "synfuels-corporation-cutback", "education-spending", "superfund-right-to-sue",
             "crime", "duty-free-exports", "export-administration-act-south-africa"]
vote_cat_cols = vote_cols[1:]
house_df = pd.DataFrame(load_vote("house-votes-84.data"), columns=vote_cols)
house_df = house_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
res_vote = run_categorical_classification("Congressional Vote", house_df, vote_cat_cols, "party", K_GRID)

print("Running Abalone...")
abalone_cols = ["sex", "length", "diameter", "height", "whole_weight", "shucked_weight",
                "viscera_weight", "shell_weight", "rings"]
abalone = pd.read_csv("abalone.data", header=None, names=abalone_cols)
abalone = pd.get_dummies(abalone, columns=["sex"])
sex_cols = [c for c in abalone.columns if c.startswith("sex_")]
abalone[sex_cols] = abalone[sex_cols].astype(float)
abalone_cols_norm = ["length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight"]
abalone_feature_cols = abalone_cols_norm + sex_cols
abalone = abalone.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
res_abalone = run_numeric_regression("Abalone", abalone, abalone_feature_cols, "rings", K_GRID, GAMMA_GRID)

print("Running Computer Hardware...")
machine_cols = ["vendor_name", "model_name", "myct", "mmin", "mmax", "cach", "chmin", "chmax", "prp", "erp"]
machine = pd.read_csv("machine.data", header=None, names=machine_cols)
machine = machine.drop(columns=["vendor_name", "model_name", "erp"])
machine_cols_norm = ["myct", "mmin", "mmax", "cach", "chmin", "chmax"]
machine = machine.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
res_machine = run_numeric_regression("Computer Hardware", machine, machine_cols_norm, "prp", K_GRID, GAMMA_GRID)

print("Running Forest Fires...")
month_order = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
day_order = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
fires = pd.read_csv("forestfires.csv")
fires["area"] = np.log1p(fires["area"])
fires["month"] = fires["month"].apply(lambda x: month_order.index(x))
fires["day"] = fires["day"].apply(lambda x: day_order.index(x))
fires_cols_norm = ["X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]
fires_cols = ["X", "Y", "month", "day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]
fires_cyclical_cols = ["month", "day"]
fires_cycle_lengths = {"month": 12, "day": 7}
fires = fires.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
res_fires = run_fires_regression("Forest Fires", fires, fires_cols, fires_cyclical_cols, fires_cycle_lengths,
                                  fires_cols_norm, "area", K_GRID, GAMMA_GRID)

ALL_RESULTS = [res_cancer, res_car, res_vote, res_abalone, res_machine, res_fires]

# 1. Aggregate results into a table

rows = []
for r in ALL_RESULTS:
    metric = "error_rate" if r["task"] == "classification" else "MSE"
    rows.append({"dataset": r["name"], "task": r["task"], "method": "Null", "metric": metric,
                 "mean_score": r["null_mean"], "hyperparams": "", "kept_fraction": np.nan,
                 "p_value_vs_previous": np.nan, "significant_at_0.05": ""})
    rows.append({"dataset": r["name"], "task": r["task"], "method": "KNN", "metric": metric,
                 "mean_score": r["knn_mean"], "hyperparams": str(r["best_params"]), "kept_fraction": np.nan,
                 "p_value_vs_previous": r["p_null_vs_knn"], "significant_at_0.05": r["p_null_vs_knn"] < 0.05})
    rows.append({"dataset": r["name"], "task": r["task"], "method": "Edited-KNN", "metric": metric,
                 "mean_score": r["edited_mean"], "hyperparams": str(r["best_params"]),
                 "kept_fraction": r["kept_fraction"],
                 "p_value_vs_previous": r["p_knn_vs_edited"], "significant_at_0.05": r["p_knn_vs_edited"] < 0.05})

results_table = pd.DataFrame(rows)
results_table.to_csv(os.path.join(OUT_DIR, "results_table.csv"), index=False)
pd.set_option("display.width", 140)
print("\n" + "=" * 90)
print("RESULTS TABLE (also written to results_table.csv)")
print("=" * 90)
print(results_table.to_string(index=False))

# 2a. Figure: validation error/MSE vs. k (all six datasets)

fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=SURFACE)
class_results = [res_cancer, res_car, res_vote]
reg_results = [res_abalone, res_machine, res_fires]

for ax, r in zip(axes[0], class_results):
    ks = [k for k, _ in r["tuning_results"]]
    errs = [e for _, e in r["tuning_results"]]
    ax.plot(ks, errs, "-o", color=C_BLUE, linewidth=2, markersize=5, zorder=3)
    best_k = r["best_params"]["k"]
    best_err = dict(r["tuning_results"])[best_k]
    ax.scatter([best_k], [best_err], color=C_ORANGE, s=90, zorder=4, edgecolor=INK, linewidth=0.5)
    ax.annotate(f"best k={best_k}", (best_k, best_err), textcoords="offset points",
                xytext=(8, 8), fontsize=8, color=INK2)
    style_axes(ax, title=r["name"], xlabel="k", ylabel="inner-CV error rate")

for ax, r in zip(axes[1], reg_results):
    by_gamma = {}
    for params, mse in r["tuning_results"]:
        by_gamma.setdefault(params["gamma"], []).append((params["k"], mse))
    for gi, gamma in enumerate(sorted(by_gamma)):
        pts = sorted(by_gamma[gamma])
        ks = [p[0] for p in pts]
        mses = [p[1] for p in pts]
        ax.plot(ks, mses, "-o", color=GAMMA_COLORS[gi % len(GAMMA_COLORS)], linewidth=2,
                markersize=4, label=f"gamma={gamma}", zorder=3)
    style_axes(ax, title=r["name"], xlabel="k", ylabel="inner-CV MSE")
    ax.legend(frameon=False, fontsize=7, labelcolor=INK2, loc="best")

fig.suptitle("Hyperparameter tuning: validation error vs. k", color=INK, fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, "fig_hyperparameter_tuning.png"), dpi=150, facecolor=SURFACE)
plt.close(fig)

# 2b. Figure: method comparison across all six datasets

fig, ax = plt.subplots(figsize=(11, 6), facecolor=SURFACE)
names = [r["name"] for r in ALL_RESULTS]
knn_rel = [r["knn_mean"] / r["null_mean"] for r in ALL_RESULTS]
ed_rel = [r["edited_mean"] / r["null_mean"] for r in ALL_RESULTS]
x = np.arange(len(names))
width = 0.35
ax.bar(x - width / 2, knn_rel, width, label="KNN", color=C_BLUE, zorder=3)
ax.bar(x + width / 2, ed_rel, width, label="Edited-KNN", color=C_ORANGE, zorder=3)
ax.axhline(1.0, color=C_GRAY, linestyle="--", linewidth=1.2, zorder=2)
ax.text(len(names) - 0.5, 1.02, "Null model baseline", color=INK2, fontsize=8, ha="right")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha="right", color=INK, fontsize=9)
style_axes(ax, title="Error relative to the null model (lower is better)",
           ylabel="mean score / null-model mean score")
ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_method_comparison.png"), dpi=150, facecolor=SURFACE)
plt.close(fig)

# 2c. Figure: reduced dataset size vs. performance (Edited-KNN)

fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
label_offsets = [(40, -25), (40, 0), (40, 25), (7, 5), (7, 5), (7, 5)]
for r, offset in zip(ALL_RESULTS, label_offsets):
    color = C_BLUE if r["task"] == "classification" else C_ORANGE
    x_val = r["kept_fraction"] * 100
    y_val = r["edited_mean"] / r["knn_mean"]
    ax.scatter([x_val], [y_val], color=color, s=110, zorder=3, edgecolor=INK, linewidth=0.5)
    ax.annotate(r["name"], (x_val, y_val), textcoords="offset points", xytext=offset,
                fontsize=8, color=INK2)
ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 10)
ax.axhline(1.0, color=C_GRAY, linestyle="--", linewidth=1.2, zorder=2)
ax.text(ax.get_xlim()[0], 1.02, "Same error as full KNN", color=INK2, fontsize=8, ha="left")
style_axes(ax, title="Edited-KNN: reduced set size vs. performance",
           xlabel="% of training points kept after editing",
           ylabel="Edited-KNN score / full-KNN score")
handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_BLUE, markersize=9, label="Classification"),
           plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_ORANGE, markersize=9, label="Regression")]
ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK2, loc="best")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_reduced_size_vs_performance.png"), dpi=150, facecolor=SURFACE)
plt.close(fig)

# 4. Statistical significance summary

print("\n" + "=" * 90)
print("SIGNIFICANCE TESTS (Dietterich 5x2cv paired t-test)")
print("=" * 90)
for r in ALL_RESULTS:
    sig1 = "significant" if r["p_null_vs_knn"] < 0.05 else "NOT significant"
    sig2 = "significant" if r["p_knn_vs_edited"] < 0.05 else "NOT significant"
    print(f"{r['name']:20s} Null vs KNN:        p={r['p_null_vs_knn']:.4f} ({sig1})")
    print(f"{'':20s} KNN vs Edited-KNN:  p={r['p_knn_vs_edited']:.4f} ({sig2})")

print("\nDone. Wrote results_table.csv, fig_hyperparameter_tuning.png, "
      "fig_method_comparison.png, fig_reduced_size_vs_performance.png")
