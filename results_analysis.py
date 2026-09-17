
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from knn import Features, KNNClassifier, KNNRegressor
from edited_condensed_knn import (
    edited_nn_classification, edited_nn_regression,
    condensed_nn_classification, condensed_nn_regression,
)
from Hyperparameter import tune_hyperparameters, classification_error, mean_squared_error

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
OUT_DIR = os.path.dirname(os.path.abspath(__file__)) or "."
EPSILON_FRACTION = 0.25  

# Palette / chart styling

C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_MAGENTA, C_GREEN, C_VIOLET, C_RED = "#e87ba4", "#008300", "#4a3aa7", "#e34948"
C_GRAY = "#898781"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
GAMMA_COLORS = [C_AQUA, C_YELLOW, C_VIOLET, C_RED]  # fixed order, distinct from method colors


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

# Small helpers

def makefeatures(df, numeric_cols, categorical_cols, cyclical_cols=None, cyc_periods=None):
    if cyclical_cols is None:
        cyclical_cols = []
    if cyc_periods is None:
        cyc_periods = []
    X_num = df[numeric_cols].to_numpy(dtype=float) if numeric_cols else np.empty((len(df), 0))
    X_cat = df[categorical_cols].to_numpy() if categorical_cols else np.empty((len(df), 0), dtype=object)
    X_cyc = df[cyclical_cols].to_numpy(dtype=float) if cyclical_cols else np.empty((len(df), 0))
    return Features(X_num=X_num, X_cat=X_cat, X_cyc=X_cyc, cyc_periods=cyc_periods)


def min_max_normalize_train_test(train_df, test_df, cols):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for col in cols:
        min_val = train_df[col].min()
        max_val = train_df[col].max()
        rng = max_val - min_val if max_val != min_val else 1.0
        train_df[col] = (train_df[col] - min_val) / rng
        test_df[col] = (test_df[col] - min_val) / rng
    return train_df, test_df


def load_vote(file_path):
    house = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split(",")
            fields = ["abstain" if val == "?" else val for val in fields]
            house.append(fields)
    return house


def classification_null_model(y_train):
    import random
    counts = y_train.value_counts()
    tied_classes = counts[counts == counts.max()].index.tolist()
    return random.choice(tied_classes)


def regression_null_model(y_train):
    return y_train.mean()


def five_by_two_split(n, random_state=None):
    """Outer 5x2cv split generator (Dietterich 1998): 5 repetitions, each
    splitting n rows into two equal-ish random halves, each used as the
    test set once. Returns 5 ((train_idx, test_idx), (train_idx, test_idx))
    pairs, i.e. 10 folds total, grouped by repetition."""
    rng = np.random.default_rng(random_state)
    reps = []
    for _ in range(5):
        idx = rng.permutation(n)
        half = n // 2
        a, b = idx[:half], idx[half:]
        reps.append(((a, b), (b, a)))
    return reps


def dietterich_5x2cv_ttest(scores_a, scores_b):
    """Paired t-test for 5x2cv (Dietterich 1998). scores_a/scores_b must
    each be a (5, 2) array from the SAME splits. Returns (t_stat, p_value, dof)."""
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    p = scores_a - scores_b
    p_bar = p.mean(axis=1)
    s2 = (p[:, 0] - p_bar) ** 2 + (p[:, 1] - p_bar) ** 2
    denom = np.sqrt(np.mean(s2))
    t_stat = 0.0 if denom == 0 else p[0, 0] / denom
    dof = 5
    p_value = 2 * stats.t.sf(np.abs(t_stat), dof)
    return float(t_stat), float(p_value), dof


def knn_classifier_factory(k, p):
    def fit_predict(train_feats, train_y, val_feats):
        model = KNNClassifier(k=k, p=p)
        model.fit(train_feats, train_y)
        return model.predict(val_feats)
    return fit_predict


def knn_regressor_factory(k, p, gamma):
    def fit_predict(train_feats, train_y, val_feats):
        model = KNNRegressor(k=k, p=p, gamma=gamma)
        model.fit(train_feats, train_y)
        return model.predict(val_feats)
    return fit_predict

# Per-dataset-shape runners: 5x2cv over Null / KNN / Edited-KNN / Condensed-KNN

def run_numeric_classification(name, df, feature_cols, class_col, k_grid, p_grid=(1, 2), inner_k=5):
    n = len(df)
    splits = five_by_two_split(n, random_state=RANDOM_STATE)
    null_s, knn_s, ed_s, cond_s = np.zeros((5, 2)), np.zeros((5, 2)), np.zeros((5, 2)), np.zeros((5, 2))
    ed_fracs, cond_fracs = [], []
    tuning_results, best_params_used = None, None

    for rep, (fold_a, fold_b) in enumerate(splits):
        for fi, (tr_idx, te_idx) in enumerate([fold_a, fold_b]):
            tr_df, te_df = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
            tr_df, te_df = min_max_normalize_train_test(tr_df, te_df, feature_cols)
            tr_y, te_y = tr_df[class_col].to_numpy(), te_df[class_col].to_numpy()

            null_pred = classification_null_model(tr_df[class_col])
            null_s[rep, fi] = classification_error(te_y, np.array([null_pred] * len(te_y)))

            train_feats = makefeatures(tr_df, feature_cols, [])
            test_feats = makefeatures(te_df, feature_cols, [])

            param_grid = {"k": list(k_grid), "p": list(p_grid)}
            best_params, _, this_tuning = tune_hyperparameters(
                knn_classifier_factory, train_feats, tr_y, param_grid,
                metric=classification_error, lower_is_better=True,
                inner_k=inner_k, random_state=RANDOM_STATE)
            if tuning_results is None:
                tuning_results = sorted((p["k"], s) for p, s in this_tuning if p["p"] == best_params["p"])
                best_params_used = best_params

            knn_model = KNNClassifier(k=best_params["k"], p=best_params["p"])
            knn_model.fit(train_feats, tr_y)
            knn_s[rep, fi] = classification_error(te_y, knn_model.predict(test_feats))

            ed_feats, ed_y, ed_keep = edited_nn_classification(train_feats, tr_y, p=best_params["p"], random_state=0)
            ed_fracs.append(ed_keep.mean())
            ed_model = KNNClassifier(k=best_params["k"], p=best_params["p"])
            ed_model.fit(ed_feats, ed_y)
            ed_s[rep, fi] = classification_error(te_y, ed_model.predict(test_feats))

            cond_feats, cond_y, cond_keep = condensed_nn_classification(train_feats, tr_y, p=best_params["p"], random_state=0)
            cond_fracs.append(cond_keep.mean())
            cond_model = KNNClassifier(k=best_params["k"], p=best_params["p"])
            cond_model.fit(cond_feats, cond_y)
            cond_s[rep, fi] = classification_error(te_y, cond_model.predict(test_feats))

    return _package_result(name, "classification", best_params_used, tuning_results,
                            null_s, knn_s, ed_s, cond_s, ed_fracs, cond_fracs)


def run_categorical_classification(name, df, cat_cols, class_col, k_grid, p_grid=(1, 2), inner_k=5):
    n = len(df)
    splits = five_by_two_split(n, random_state=RANDOM_STATE)
    null_s, knn_s, ed_s, cond_s = np.zeros((5, 2)), np.zeros((5, 2)), np.zeros((5, 2)), np.zeros((5, 2))
    ed_fracs, cond_fracs = [], []
    tuning_results, best_params_used = None, None

    for rep, (fold_a, fold_b) in enumerate(splits):
        for fi, (tr_idx, te_idx) in enumerate([fold_a, fold_b]):
            tr_df, te_df = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
            tr_y, te_y = tr_df[class_col].to_numpy(), te_df[class_col].to_numpy()

            null_pred = classification_null_model(tr_df[class_col])
            null_s[rep, fi] = classification_error(te_y, np.array([null_pred] * len(te_y)))

            train_feats = makefeatures(tr_df, [], cat_cols)
            test_feats = makefeatures(te_df, [], cat_cols)

            param_grid = {"k": list(k_grid), "p": list(p_grid)}
            best_params, _, this_tuning = tune_hyperparameters(
                knn_classifier_factory, train_feats, tr_y, param_grid,
                metric=classification_error, lower_is_better=True,
                inner_k=inner_k, random_state=RANDOM_STATE)
            if tuning_results is None:
                tuning_results = sorted((p["k"], s) for p, s in this_tuning if p["p"] == best_params["p"])
                best_params_used = best_params

            knn_model = KNNClassifier(k=best_params["k"], p=best_params["p"])
            knn_model.fit(train_feats, tr_y)
            knn_s[rep, fi] = classification_error(te_y, knn_model.predict(test_feats))

            ed_feats, ed_y, ed_keep = edited_nn_classification(train_feats, tr_y, p=best_params["p"], random_state=0)
            ed_fracs.append(ed_keep.mean())
            ed_model = KNNClassifier(k=best_params["k"], p=best_params["p"])
            ed_model.fit(ed_feats, ed_y)
            ed_s[rep, fi] = classification_error(te_y, ed_model.predict(test_feats))

            cond_feats, cond_y, cond_keep = condensed_nn_classification(train_feats, tr_y, p=best_params["p"], random_state=0)
            cond_fracs.append(cond_keep.mean())
            cond_model = KNNClassifier(k=best_params["k"], p=best_params["p"])
            cond_model.fit(cond_feats, cond_y)
            cond_s[rep, fi] = classification_error(te_y, cond_model.predict(test_feats))

    return _package_result(name, "classification", best_params_used, tuning_results,
                            null_s, knn_s, ed_s, cond_s, ed_fracs, cond_fracs)


def run_numeric_regression(name, df, feature_cols, target_col, k_grid, gamma_grid, p_grid=(2,), inner_k=5):
    n = len(df)
    splits = five_by_two_split(n, random_state=RANDOM_STATE)
    null_s, knn_s, ed_s, cond_s = np.zeros((5, 2)), np.zeros((5, 2)), np.zeros((5, 2)), np.zeros((5, 2))
    ed_fracs, cond_fracs = [], []
    tuning_results, best_params_used = None, None

    for rep, (fold_a, fold_b) in enumerate(splits):
        for fi, (tr_idx, te_idx) in enumerate([fold_a, fold_b]):
            tr_df, te_df = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
            tr_df, te_df = min_max_normalize_train_test(tr_df, te_df, feature_cols)
            tr_y, te_y = tr_df[target_col].to_numpy(dtype=float), te_df[target_col].to_numpy(dtype=float)

            null_pred = regression_null_model(tr_df[target_col])
            null_s[rep, fi] = mean_squared_error(te_y, np.full(len(te_y), null_pred))

            train_feats = makefeatures(tr_df, feature_cols, [])
            test_feats = makefeatures(te_df, feature_cols, [])

            param_grid = {"k": list(k_grid), "p": list(p_grid), "gamma": list(gamma_grid)}
            best_params, _, this_tuning = tune_hyperparameters(
                knn_regressor_factory, train_feats, tr_y, param_grid,
                metric=mean_squared_error, lower_is_better=True,
                inner_k=inner_k, random_state=RANDOM_STATE)
            if tuning_results is None:
                tuning_results = this_tuning
                best_params_used = best_params

            knn_model = KNNRegressor(k=best_params["k"], p=best_params["p"], gamma=best_params["gamma"])
            knn_model.fit(train_feats, tr_y)
            knn_s[rep, fi] = mean_squared_error(te_y, knn_model.predict(test_feats))

            epsilon = EPSILON_FRACTION * np.std(tr_y)

            ed_feats, ed_y, ed_keep = edited_nn_regression(
                train_feats, tr_y, epsilon=epsilon, p=best_params["p"], gamma=best_params["gamma"], random_state=0)
            ed_fracs.append(ed_keep.mean())
            ed_model = KNNRegressor(k=best_params["k"], p=best_params["p"], gamma=best_params["gamma"])
            ed_model.fit(ed_feats, ed_y)
            ed_s[rep, fi] = mean_squared_error(te_y, ed_model.predict(test_feats))

            cond_feats, cond_y, cond_keep = condensed_nn_regression(
                train_feats, tr_y, epsilon=epsilon, p=best_params["p"], gamma=best_params["gamma"], random_state=0)
            cond_fracs.append(cond_keep.mean())
            cond_model = KNNRegressor(k=best_params["k"], p=best_params["p"], gamma=best_params["gamma"])
            cond_model.fit(cond_feats, cond_y)
            cond_s[rep, fi] = mean_squared_error(te_y, cond_model.predict(test_feats))

    return _package_result(name, "regression", best_params_used, tuning_results,
                            null_s, knn_s, ed_s, cond_s, ed_fracs, cond_fracs)


def run_fires_regression(name, df, cols, cyclical_cols, cycle_lengths, norm_cols, target_col,
                          k_grid, gamma_grid, p_grid=(2,), inner_k=5):
    cyc_periods = [cycle_lengths[c] for c in cyclical_cols]
    plain_cols = [c for c in cols if c not in cyclical_cols]

    n = len(df)
    splits = five_by_two_split(n, random_state=RANDOM_STATE)
    null_s, knn_s, ed_s, cond_s = np.zeros((5, 2)), np.zeros((5, 2)), np.zeros((5, 2)), np.zeros((5, 2))
    ed_fracs, cond_fracs = [], []
    tuning_results, best_params_used = None, None

    for rep, (fold_a, fold_b) in enumerate(splits):
        for fi, (tr_idx, te_idx) in enumerate([fold_a, fold_b]):
            tr_df, te_df = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
            tr_df, te_df = min_max_normalize_train_test(tr_df, te_df, norm_cols)
            tr_y, te_y = tr_df[target_col].to_numpy(dtype=float), te_df[target_col].to_numpy(dtype=float)

            null_pred = regression_null_model(tr_df[target_col])
            null_s[rep, fi] = mean_squared_error(te_y, np.full(len(te_y), null_pred))

            train_feats = makefeatures(tr_df, plain_cols, [], cyclical_cols, cyc_periods)
            test_feats = makefeatures(te_df, plain_cols, [], cyclical_cols, cyc_periods)

            param_grid = {"k": list(k_grid), "p": list(p_grid), "gamma": list(gamma_grid)}
            best_params, _, this_tuning = tune_hyperparameters(
                knn_regressor_factory, train_feats, tr_y, param_grid,
                metric=mean_squared_error, lower_is_better=True,
                inner_k=inner_k, random_state=RANDOM_STATE)
            if tuning_results is None:
                tuning_results = this_tuning
                best_params_used = best_params

            knn_model = KNNRegressor(k=best_params["k"], p=best_params["p"], gamma=best_params["gamma"])
            knn_model.fit(train_feats, tr_y)
            knn_s[rep, fi] = mean_squared_error(te_y, knn_model.predict(test_feats))

            epsilon = EPSILON_FRACTION * np.std(tr_y)

            ed_feats, ed_y, ed_keep = edited_nn_regression(
                train_feats, tr_y, epsilon=epsilon, p=best_params["p"], gamma=best_params["gamma"], random_state=0)
            ed_fracs.append(ed_keep.mean())
            ed_model = KNNRegressor(k=best_params["k"], p=best_params["p"], gamma=best_params["gamma"])
            ed_model.fit(ed_feats, ed_y)
            ed_s[rep, fi] = mean_squared_error(te_y, ed_model.predict(test_feats))

            cond_feats, cond_y, cond_keep = condensed_nn_regression(
                train_feats, tr_y, epsilon=epsilon, p=best_params["p"], gamma=best_params["gamma"], random_state=0)
            cond_fracs.append(cond_keep.mean())
            cond_model = KNNRegressor(k=best_params["k"], p=best_params["p"], gamma=best_params["gamma"])
            cond_model.fit(cond_feats, cond_y)
            cond_s[rep, fi] = mean_squared_error(te_y, cond_model.predict(test_feats))

    return _package_result(name, "regression", best_params_used, tuning_results,
                            null_s, knn_s, ed_s, cond_s, ed_fracs, cond_fracs)


def _package_result(name, task, best_params, tuning_results, null_s, knn_s, ed_s, cond_s, ed_fracs, cond_fracs):
    _, p_null_knn, _ = dietterich_5x2cv_ttest(null_s, knn_s)
    _, p_knn_ed, _ = dietterich_5x2cv_ttest(knn_s, ed_s)
    _, p_ed_cond, _ = dietterich_5x2cv_ttest(ed_s, cond_s)
    _, p_knn_cond, _ = dietterich_5x2cv_ttest(knn_s, cond_s)
    return {
        "name": name, "task": task, "best_params": best_params, "tuning_results": tuning_results,
        "null_mean": float(null_s.mean()), "knn_mean": float(knn_s.mean()),
        "edited_mean": float(ed_s.mean()), "condensed_mean": float(cond_s.mean()),
        "edited_kept_fraction": float(np.mean(ed_fracs)), "condensed_kept_fraction": float(np.mean(cond_fracs)),
        "p_null_vs_knn": p_null_knn, "p_knn_vs_edited": p_knn_ed,
        "p_edited_vs_condensed": p_ed_cond, "p_knn_vs_condensed": p_knn_cond,
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
                 "kept_fraction": r["edited_kept_fraction"],
                 "p_value_vs_previous": r["p_knn_vs_edited"], "significant_at_0.05": r["p_knn_vs_edited"] < 0.05})
    rows.append({"dataset": r["name"], "task": r["task"], "method": "Condensed-KNN", "metric": metric,
                 "mean_score": r["condensed_mean"], "hyperparams": str(r["best_params"]),
                 "kept_fraction": r["condensed_kept_fraction"],
                 "p_value_vs_previous": r["p_edited_vs_condensed"],
                 "significant_at_0.05": r["p_edited_vs_condensed"] < 0.05})

results_table = pd.DataFrame(rows)
results_table.to_csv(os.path.join(OUT_DIR, "results_table.csv"), index=False)
pd.set_option("display.width", 140)
print("\n" + "=" * 90)
print("RESULTS TABLE (also written to results_table.csv)")
print("=" * 90)
print(results_table.to_string(index=False))

# 2a. Figure: validation error/MSE vs. k

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

fig, ax = plt.subplots(figsize=(12, 6), facecolor=SURFACE)
names = [r["name"] for r in ALL_RESULTS]
knn_rel = [r["knn_mean"] / r["null_mean"] for r in ALL_RESULTS]
ed_rel = [r["edited_mean"] / r["null_mean"] for r in ALL_RESULTS]
cond_rel = [r["condensed_mean"] / r["null_mean"] for r in ALL_RESULTS]
x = np.arange(len(names))
width = 0.26
ax.bar(x - width, knn_rel, width, label="KNN", color=C_BLUE, zorder=3)
ax.bar(x, ed_rel, width, label="Edited-KNN", color=C_ORANGE, zorder=3)
ax.bar(x + width, cond_rel, width, label="Condensed-KNN", color=C_AQUA, zorder=3)
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

# 2c. Figure: reduced dataset size vs. performance

DATASET_COLORS = {
    "Breast Cancer": C_BLUE, "Car Evaluation": C_ORANGE, "Congressional Vote": C_AQUA,
    "Abalone": C_YELLOW, "Computer Hardware": C_MAGENTA, "Forest Fires": C_VIOLET,
}
fig, ax = plt.subplots(figsize=(9, 7), facecolor=SURFACE)
for r in ALL_RESULTS:
    color = DATASET_COLORS[r["name"]]
    ax.scatter([r["edited_kept_fraction"] * 100], [r["edited_mean"] / r["knn_mean"]],
               color=color, s=130, marker="o", edgecolor=INK, linewidth=0.6, zorder=3)
    ax.scatter([r["condensed_kept_fraction"] * 100], [r["condensed_mean"] / r["knn_mean"]],
               color=color, s=130, marker="^", edgecolor=INK, linewidth=0.6, zorder=3)
ax.axhline(1.0, color=C_GRAY, linestyle="--", linewidth=1.2, zorder=2)
ax.text(55, 1.03, "Same error as full KNN", color=INK2, fontsize=8, ha="center")
style_axes(ax, title="Reduced set size vs. performance (Edited-KNN vs. Condensed-KNN)",
           xlabel="% of training points kept after reduction",
           ylabel="reduced-set score / full-KNN score")
dataset_handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=9, label=name)
                   for name, c in DATASET_COLORS.items()]
method_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_GRAY, markersize=9, label="Edited-KNN"),
                  plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=C_GRAY, markersize=9, label="Condensed-KNN")]
leg1 = ax.legend(handles=dataset_handles, loc="lower left", frameon=False, fontsize=8,
                  labelcolor=INK2, title="Dataset", title_fontsize=8)
leg1.get_title().set_color(INK2)
ax.add_artist(leg1)
leg2 = ax.legend(handles=method_handles, loc="upper right", frameon=False, fontsize=8,
                  labelcolor=INK2, title="Method", title_fontsize=8)
leg2.get_title().set_color(INK2)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_reduced_size_vs_performance.png"), dpi=150, facecolor=SURFACE)
plt.close(fig)

# 4. Statistical significance summary

print("\n" + "=" * 90)
print("SIGNIFICANCE TESTS (Dietterich 5x2cv paired t-test)")
print("=" * 90)
def _fmt(p):
    return "significant" if p < 0.05 else "NOT significant"

for r in ALL_RESULTS:
    print(f"{r['name']:20s} Null vs KNN:            p={r['p_null_vs_knn']:.4f} ({_fmt(r['p_null_vs_knn'])})")
    print(f"{'':20s} KNN vs Edited-KNN:      p={r['p_knn_vs_edited']:.4f} ({_fmt(r['p_knn_vs_edited'])})")
    print(f"{'':20s} KNN vs Condensed-KNN:   p={r['p_knn_vs_condensed']:.4f} ({_fmt(r['p_knn_vs_condensed'])})")
    print(f"{'':20s} Edited vs Condensed-KNN: p={r['p_edited_vs_condensed']:.4f} ({_fmt(r['p_edited_vs_condensed'])})")

print("\nDone. Wrote results_table.csv, fig_hyperparameter_tuning.png, "
      "fig_method_comparison.png, fig_reduced_size_vs_performance.png")
