
import random
import pandas as pd
import numpy as np

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Normalization

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
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split(',')
            fields = ['abstain' if val == '?' else val for val in fields]
            house.append(fields)
    return house

# Distance functions

def minkowski_distance(x, y, p):
    return np.sum(np.abs(x - y) ** p) ** (1 / p)


def compute_vdm_tables(train_df, cat_col, class_col):
    counts = {}
    classes = train_df[class_col].unique()
    for col in train_df[cat_col].unique():
        subset = train_df[train_df[cat_col] == col]
        Ci = len(subset)
        Ci_a = {c: len(subset[subset[class_col] == c]) for c in classes}
        counts[col] = {'Ci': Ci, 'Ci_a': Ci_a}
    return counts


def vdm_delta(vi, vj, vdm_table, classes, p=1):
    if vi == vj:
        return 0.0
    Ci = vdm_table[vi]['Ci']
    Cj = vdm_table[vj]['Ci']
    total = 0.0
    for c in classes:
        Ci_a = vdm_table[vi]['Ci_a'][c]
        Cj_a = vdm_table[vj]['Ci_a'][c]
        total += np.abs((Ci_a / Ci) - (Cj_a / Cj)) ** p
    return total


def cat_dist(x_cat, y_cat, vdm_tables, classes, p=1):
    dist = 0.0
    for k in range(len(x_cat)):
        dist += vdm_delta(x_cat[k], y_cat[k], vdm_tables[k], classes, p=p)
    return dist ** (1 / p)


def cyclical_distance(a, b, cycle_length):
    diff = abs(a - b)
    return min(diff, cycle_length - diff)


def fires_distance(x, y, cols, cyclical_cols, cycle_lengths, p=2):
    total = 0.0
    for col in cols:
        if col in cyclical_cols:
            diff = cyclical_distance(x[col], y[col], cycle_lengths[col])
        else:
            diff = abs(x[col] - y[col])
        total += diff ** p
    return total ** (1 / p)

# Metrics

def classification_error(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    return (1 / len(y_true)) * np.sum(y_pred != y_true)


def mean_squared_error(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return (1 / len(y_true)) * np.sum((y_pred - y_true) ** 2)   # <-- fixed

# Null models

def classification_null_model(y_train):
    counts = y_train.value_counts()
    tied_classes = counts[counts == counts.max()].index.tolist()
    return random.choice(tied_classes)


def regression_null_model(y_train):
    return y_train.mean()

# Pairwise distance matrix builders 

def pairwise_numeric(A, B, p):
    """A, B: 2D numpy arrays of numeric columns. Returns (len(A), len(B))
    of Minkowski distances -- same math as minkowski_distance, vectorized
    across all of B for each row of A."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    D = np.zeros((len(A), len(B)))
    for i in range(len(A)):
        D[i] = np.sum(np.abs(B - A[i]) ** p, axis=1) ** (1.0 / p)
    return D


def pairwise_categorical(A_cat, B_cat, vdm_tables, classes, p=1):
    """A_cat, B_cat: lists of rows (each row a list of category values, in
    the same column order used to build vdm_tables). Calls cat_dist for
    every pair."""
    n_a, n_b = len(A_cat), len(B_cat)
    D = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            D[i, j] = cat_dist(A_cat[i], B_cat[j], vdm_tables, classes, p=p)
    return D


def pairwise_fires(A_rows, B_rows, cols, cyclical_cols, cycle_lengths, p=2):
    """A_rows, B_rows: lists of dict-like rows (df.to_dict('records')).
    Calls fires_distance for every pair."""
    n_a, n_b = len(A_rows), len(B_rows)
    D = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            D[i, j] = fires_distance(A_rows[i], B_rows[j], cols, cyclical_cols, cycle_lengths, p=p)
    return D

# Plain KNN

def knn_classify_row(distances_row, train_y, k):
    idx = np.argsort(distances_row, kind='stable')[:k]
    neighbor_labels = np.asarray(train_y)[idx]
    values, counts = np.unique(neighbor_labels, return_counts=True)
    top = values[counts == counts.max()]
    return random.choice(list(top))


def knn_classify(D, train_y, k):
    return np.array([knn_classify_row(D[i], train_y, k) for i in range(D.shape[0])])


def knn_regress_row(distances_row, train_y, k, gamma):
    idx = np.argsort(distances_row, kind='stable')[:k]
    d = distances_row[idx]
    y_neighbors = np.asarray(train_y, dtype=float)[idx]
    weights = np.exp(-gamma * d)
    if weights.sum() <= 1e-12:
        return y_neighbors.mean()
    return np.sum(weights * y_neighbors) / weights.sum()


def knn_regress(D, train_y, k, gamma):
    return np.array([knn_regress_row(D[i], train_y, k, gamma) for i in range(D.shape[0])])

# Edited-KNN (Wilson editing)

def edited_nn_classification(D_full, y, max_iters=50, min_class_count=1, random_state=None):
    rng = random.Random(random_state)
    y = np.asarray(y)
    n = len(y)
    keep = np.ones(n, dtype=bool)

    for _ in range(max_iters):
        idx = np.where(keep)[0]
        if len(idx) <= 2:
            break
        sub_D = D_full[np.ix_(idx, idx)].copy()
        np.fill_diagonal(sub_D, np.inf)
        nn_local = np.argmin(sub_D, axis=1)
        loo_pred = y[idx][nn_local]
        wrong = loo_pred != y[idx]

        removable = np.ones(len(idx), dtype=bool)
        for cls in np.unique(y[idx]):
            cls_mask = y[idx] == cls
            n_would_keep = (cls_mask & ~wrong).sum()
            if n_would_keep < min_class_count:
                wrong_cls_idx = np.where(wrong & cls_mask)[0].tolist()
                spare = rng.sample(wrong_cls_idx, k=min(min_class_count - n_would_keep, len(wrong_cls_idx)))
                removable[spare] = False

        to_remove = wrong & removable
        if not to_remove.any():
            break
        keep[idx[to_remove]] = False

    return keep


def edited_nn_regression(D_full, y, epsilon, max_iters=50, min_remaining=5):
    y = np.asarray(y, dtype=float)
    n = len(y)
    keep = np.ones(n, dtype=bool)

    for _ in range(max_iters):
        idx = np.where(keep)[0]
        if len(idx) <= min_remaining:
            break
        sub_D = D_full[np.ix_(idx, idx)].copy()
        np.fill_diagonal(sub_D, np.inf)
        nn_local = np.argmin(sub_D, axis=1)
        loo_pred = y[idx][nn_local]
        errors = np.abs(loo_pred - y[idx])
        wrong = errors > epsilon
        if not wrong.any():
            break

        n_would_remain = len(idx) - wrong.sum()
        if n_would_remain < min_remaining:
            budget = len(idx) - min_remaining
            if budget <= 0:
                break
            worst_order = np.argsort(-errors)
            chosen = [i for i in worst_order if wrong[i]][:budget]
            mask = np.zeros(len(idx), dtype=bool)
            mask[chosen] = True
            wrong = mask

        keep[idx[wrong]] = False

    return keep

# Hyperparameter tuning + 5x2cv significance testing

def k_fold_split(n, k, random_state=None):
    """Yields (train_idx, val_idx) index arrays for plain k-fold CV."""
    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, val_idx


def tune_k_numeric_classification(train_df, feature_cols, class_col, k_grid, p=2,
                                   inner_k=5, random_state=None):
    """Picks k by inner k-fold CV, for numeric datasets like Breast Cancer.
    Re-normalizes on each inner fold's own training slice (never leaks the
    outer training set's min/max into the inner validation rows)."""
    n = len(train_df)
    results = []
    for k in k_grid:
        fold_errors = []
        for tr_idx, val_idx in k_fold_split(n, inner_k, random_state=random_state):
            tr_df = train_df.iloc[tr_idx].copy()
            val_df = train_df.iloc[val_idx].copy()
            tr_df, val_df = min_max_normalize_train_test(tr_df, val_df, feature_cols)
            D = pairwise_numeric(val_df[feature_cols].values, tr_df[feature_cols].values, p=p)
            preds = knn_classify(D, tr_df[class_col].values, k)
            fold_errors.append(classification_error(preds, val_df[class_col].values))
        results.append((k, float(np.mean(fold_errors))))
    results.sort(key=lambda t: t[1])
    best_k, best_err = results[0]
    return best_k, best_err, results


def tune_k_categorical_classification(train_df, cat_cols, class_col, k_grid, p=1,
                                       inner_k=5, random_state=None):
    """Picks k by inner k-fold CV, for VDM/categorical datasets like Car
    Evaluation and Congressional Vote. Refits VDM tables on each inner
    fold's own training slice."""
    n = len(train_df)
    results = []
    for k in k_grid:
        fold_errors = []
        for tr_idx, val_idx in k_fold_split(n, inner_k, random_state=random_state):
            tr_df = train_df.iloc[tr_idx]
            val_df = train_df.iloc[val_idx]
            vdm_tables = [compute_vdm_tables(tr_df, col, class_col) for col in cat_cols]
            classes = tr_df[class_col].unique()
            X_tr = tr_df[cat_cols].values.tolist()
            X_val = val_df[cat_cols].values.tolist()
            D = pairwise_categorical(X_val, X_tr, vdm_tables, classes, p=p)
            preds = knn_classify(D, tr_df[class_col].values, k)
            fold_errors.append(classification_error(preds, val_df[class_col].values))
        results.append((k, float(np.mean(fold_errors))))
    results.sort(key=lambda t: t[1])
    best_k, best_err = results[0]
    return best_k, best_err, results


def tune_knn_regression_numeric(train_df, feature_cols, target_col, k_grid, gamma_grid, p=2,
                                 inner_k=5, random_state=None):
    """Picks (k, gamma) by inner k-fold CV, for numeric regression data
    sets like Abalone and Computer Hardware."""
    n = len(train_df)
    results = []
    for k in k_grid:
        for gamma in gamma_grid:
            fold_mses = []
            for tr_idx, val_idx in k_fold_split(n, inner_k, random_state=random_state):
                tr_df = train_df.iloc[tr_idx].copy()
                val_df = train_df.iloc[val_idx].copy()
                tr_df, val_df = min_max_normalize_train_test(tr_df, val_df, feature_cols)
                D = pairwise_numeric(val_df[feature_cols].values, tr_df[feature_cols].values, p=p)
                preds = knn_regress(D, tr_df[target_col].values, k, gamma)
                fold_mses.append(mean_squared_error(preds, val_df[target_col].values))
            results.append(({'k': k, 'gamma': gamma}, float(np.mean(fold_mses))))
    results.sort(key=lambda t: t[1])
    best_params, best_mse = results[0]
    return best_params, best_mse, results


def tune_knn_regression_fires(train_df, cols, cyclical_cols, cycle_lengths, norm_cols, target_col,
                               k_grid, gamma_grid, p=2, inner_k=5, random_state=None):
    """Picks (k, gamma) by inner k-fold CV, for Forest Fires (numeric +
    cyclical regression)."""
    n = len(train_df)
    results = []
    for k in k_grid:
        for gamma in gamma_grid:
            fold_mses = []
            for tr_idx, val_idx in k_fold_split(n, inner_k, random_state=random_state):
                tr_df = train_df.iloc[tr_idx].copy()
                val_df = train_df.iloc[val_idx].copy()
                tr_df, val_df = min_max_normalize_train_test(tr_df, val_df, norm_cols)
                tr_rows = tr_df[cols].to_dict('records')
                val_rows = val_df[cols].to_dict('records')
                D = pairwise_fires(val_rows, tr_rows, cols, cyclical_cols, cycle_lengths, p=p)
                preds = knn_regress(D, tr_df[target_col].values, k, gamma)
                fold_mses.append(mean_squared_error(preds, val_df[target_col].values))
            results.append(({'k': k, 'gamma': gamma}, float(np.mean(fold_mses))))
    results.sort(key=lambda t: t[1])
    best_params, best_mse = results[0]
    return best_params, best_mse, results


def five_by_two_split(n, random_state=None):
    """Outer 5x2cv split generator (Dietterich 1998): 5 repetitions, each
    splitting the n rows into two equal-ish random halves and using each
    half as the test set once. Returns a list of 5 ((idxA, idxB), (idxB,
    idxA)) pairs -- i.e. 10 (train_idx, test_idx) folds total, grouped by
    repetition so dietterich_5x2cv_ttest can pair them up correctly."""
    rng = np.random.default_rng(random_state)
    reps = []
    for _ in range(5):
        idx = rng.permutation(n)
        half = n // 2
        a, b = idx[:half], idx[half:]
        reps.append(((a, b), (b, a)))  # (train, test), (train, test)
    return reps


def dietterich_5x2cv_ttest(scores_a, scores_b):
    """Paired t-test for 5x2cv (Dietterich 1998). scores_a/scores_b must
    each be a (5, 2) array: 5 repetitions x 2 folds, aligned to the SAME
    splits (i.e. produced from the same five_by_two_split(...) call) so
    that scores_a[i, j] and scores_b[i, j] come from identical train/test
    data. Returns (t_stat, p_value, dof)."""
    from scipy import stats
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    p = scores_a - scores_b               # (5, 2) differences
    p_bar = p.mean(axis=1)                # mean diff per repetition, shape (5,)
    s2 = (p[:, 0] - p_bar) ** 2 + (p[:, 1] - p_bar) ** 2   # shape (5,)
    denom = np.sqrt(np.mean(s2))
    t_stat = 0.0 if denom == 0 else p[0, 0] / denom
    dof = 5
    p_value = 2 * stats.t.sf(np.abs(t_stat), dof)
    return float(t_stat), float(p_value), dof


def section(name):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")


if __name__ == "__main__":
    # =======================================================================
    # 1. Breast Cancer -- numeric classification
    # =======================================================================
    section("Breast Cancer")
    cancer_cols = ['id', 'clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity',
                   'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
                   'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
    cancer = pd.read_csv("breast-cancer-wisconsin.data", header=None, names=cancer_cols)
    cancer = cancer.drop(columns=['id'])
    cancer = cancer[cancer['bare_nuclei'] != '?']
    cancer['bare_nuclei'] = cancer['bare_nuclei'].astype(int)
    cancer_cols_norm = ['clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity',
                         'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
                         'bland_chromatin', 'normal_nucleoli', 'mitoses']

    cancer_shuffled = cancer.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    split_idx = int(len(cancer_shuffled) * 0.8)
    cancer_train = cancer_shuffled.iloc[:split_idx].copy()
    cancer_test = cancer_shuffled.iloc[split_idx:].copy()
    cancer_train, cancer_test = min_max_normalize_train_test(cancer_train, cancer_test, cancer_cols_norm)

    X_train_c = cancer_train[cancer_cols_norm].values
    X_test_c = cancer_test[cancer_cols_norm].values
    y_train_c = cancer_train['class'].values
    y_test_c = cancer_test['class'].values

    pred = classification_null_model(cancer_train['class'])
    print("Null model error:", classification_error([pred] * len(y_test_c), y_test_c))

    D_cancer = pairwise_numeric(X_test_c, X_train_c, p=2)
    preds_cancer = knn_classify(D_cancer, y_train_c, k=5)
    print("KNN (k=5) error:", classification_error(preds_cancer, y_test_c))

    D_cancer_train = pairwise_numeric(X_train_c, X_train_c, p=2)
    keep_c = edited_nn_classification(D_cancer_train, y_train_c, random_state=0)
    print(f"Edited set kept {keep_c.sum()} / {len(keep_c)} points")
    D_cancer_edited = pairwise_numeric(X_test_c, X_train_c[keep_c], p=2)
    preds_cancer_edited = knn_classify(D_cancer_edited, y_train_c[keep_c], k=5)
    print("Edited-KNN (k=5) error:", classification_error(preds_cancer_edited, y_test_c))


    # =======================================================================
    # 2. Car Evaluation -- categorical (VDM) classification
    # =======================================================================
    section("Car Evaluation")
    car_cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    car_cat_cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    car = pd.read_csv("car.data", header=None, names=car_cols)

    car_shuffled = car.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    split_idx = int(len(car_shuffled) * 0.8)
    car_train = car_shuffled.iloc[:split_idx].copy()
    car_test = car_shuffled.iloc[split_idx:].copy()

    car_vdm_tables = [compute_vdm_tables(car_train, col, 'class') for col in car_cat_cols]
    car_classes = car_train['class'].unique()

    y_train_car = car_train['class']
    y_test_car = car_test['class']
    pred = classification_null_model(y_train_car)
    print("Null model error:", classification_error([pred] * len(y_test_car), y_test_car))

    X_cat_train = car_train[car_cat_cols].values.tolist()
    X_cat_test = car_test[car_cat_cols].values.tolist()

    D_car = pairwise_categorical(X_cat_test, X_cat_train, car_vdm_tables, car_classes, p=1)
    preds_car = knn_classify(D_car, y_train_car.values, k=5)
    print("KNN (k=5) error:", classification_error(preds_car, y_test_car.values))

    D_car_train = pairwise_categorical(X_cat_train, X_cat_train, car_vdm_tables, car_classes, p=1)
    keep_car = edited_nn_classification(D_car_train, y_train_car.values, random_state=0)
    print(f"Edited set kept {keep_car.sum()} / {len(keep_car)} points")


    # =======================================================================
    # 3. Congressional Vote -- categorical (VDM) classification
    # =======================================================================
    section("Congressional Vote")
    vote_cols = ['party', 'handicapped-infants', 'water-project-cost-sharing',
                 'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid',
                 'religious-groups-in-schools', 'anti-satellite-test-ban',
                 'aid-to-nicaraguan-contras', 'mx-missile', 'immigration',
                 'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue',
                 'crime', 'duty-free-exports', 'export-administration-act-south-africa']
    vote_cat_cols = vote_cols[1:]
    house_data = load_vote("house-votes-84.data")
    house_df = pd.DataFrame(house_data, columns=vote_cols)

    house_shuffled = house_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    split_idx = int(len(house_shuffled) * 0.8)
    house_train = house_shuffled.iloc[:split_idx].copy()
    house_test = house_shuffled.iloc[split_idx:].copy()

    house_vdm_tables = [compute_vdm_tables(house_train, col, 'party') for col in vote_cat_cols]
    house_classes = house_train['party'].unique()

    y_train_house = house_train['party']
    y_test_house = house_test['party']
    pred = classification_null_model(y_train_house)
    print("Null model error:", classification_error([pred] * len(y_test_house), y_test_house))

    X_cat_train_h = house_train[vote_cat_cols].values.tolist()
    X_cat_test_h = house_test[vote_cat_cols].values.tolist()

    D_house = pairwise_categorical(X_cat_test_h, X_cat_train_h, house_vdm_tables, house_classes, p=1)
    preds_house = knn_classify(D_house, y_train_house.values, k=5)
    print("KNN (k=5) error:", classification_error(preds_house, y_test_house.values))

    D_house_train = pairwise_categorical(X_cat_train_h, X_cat_train_h, house_vdm_tables, house_classes, p=1)
    keep_house = edited_nn_classification(D_house_train, y_train_house.values, random_state=0)
    print(f"Edited set kept {keep_house.sum()} / {len(keep_house)} points")


    # =======================================================================
    # 4. Abalone -- numeric regression
    # =======================================================================
    section("Abalone")
    abalone_cols = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',
                    'viscera_weight', 'shell_weight', 'rings']
    abalone = pd.read_csv("abalone.data", header=None, names=abalone_cols)
    abalone = pd.get_dummies(abalone, columns=['sex'])
    sex_cols = [c for c in abalone.columns if c.startswith('sex_')]
    abalone[sex_cols] = abalone[sex_cols].astype(float)
    abalone_cols_norm = ['length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']

    abalone_shuffled = abalone.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    split_idx = int(len(abalone_shuffled) * 0.8)
    abalone_train = abalone_shuffled.iloc[:split_idx].copy()
    abalone_test = abalone_shuffled.iloc[split_idx:].copy()
    abalone_train, abalone_test = min_max_normalize_train_test(abalone_train, abalone_test, abalone_cols_norm)

    feature_cols_ab = abalone_cols_norm + sex_cols
    X_train_ab = abalone_train[feature_cols_ab].values.astype(float)
    X_test_ab = abalone_test[feature_cols_ab].values.astype(float)
    y_train_ab = abalone_train['rings'].values
    y_test_ab = abalone_test['rings'].values

    null_pred_ab = regression_null_model(abalone_train['rings'])
    print("Null model MSE:", mean_squared_error([null_pred_ab] * len(y_test_ab), y_test_ab))

    D_ab = pairwise_numeric(X_test_ab, X_train_ab, p=2)
    preds_ab = knn_regress(D_ab, y_train_ab, k=5, gamma=1.0)
    print("KNN (k=5, gamma=1.0) MSE:", mean_squared_error(preds_ab, y_test_ab))


    # =======================================================================
    # 5. Computer Hardware -- numeric regression
    # =======================================================================
    section("Computer Hardware")
    machine_cols = ['vendor_name', 'model_name', 'myct', 'mmin', 'mmax', 'cach', 'chmin', 'chmax', 'prp', 'erp']
    machine = pd.read_csv("machine.data", header=None, names=machine_cols)
    erp = machine['erp']  # kept aside, never used as a feature
    machine = machine.drop(columns=['vendor_name', 'model_name', 'erp'])
    machine_cols_norm = ['myct', 'mmin', 'mmax', 'cach', 'chmin', 'chmax']

    machine_shuffled = machine.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    split_idx = int(len(machine_shuffled) * 0.8)
    machine_train = machine_shuffled.iloc[:split_idx].copy()
    machine_test = machine_shuffled.iloc[split_idx:].copy()
    machine_train, machine_test = min_max_normalize_train_test(machine_train, machine_test, machine_cols_norm)

    X_train_m = machine_train[machine_cols_norm].values
    X_test_m = machine_test[machine_cols_norm].values
    y_train_m = machine_train['prp'].values
    y_test_m = machine_test['prp'].values

    null_pred_m = regression_null_model(machine_train['prp'])
    print("Null model MSE:", mean_squared_error([null_pred_m] * len(y_test_m), y_test_m))

    D_m = pairwise_numeric(X_test_m, X_train_m, p=2)
    preds_m = knn_regress(D_m, y_train_m, k=5, gamma=1.0)
    print("KNN (k=5, gamma=1.0) MSE:", mean_squared_error(preds_m, y_test_m))


    # =======================================================================
    # 6. Forest Fires -- numeric + cyclical regression
    # =======================================================================
    section("Forest Fires")
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    day_order = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    fires = pd.read_csv("forestfires.csv")
    fires['area'] = np.log1p(fires['area'])
    fires['month'] = fires['month'].apply(lambda x: month_order.index(x))
    fires['day'] = fires['day'].apply(lambda x: day_order.index(x))
    fires_cols_norm = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
    fires_cols = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
    cyclical_cols = ['month', 'day']
    cycle_lengths = {'month': 12, 'day': 7}

    fires_shuffled = fires.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    split_idx = int(len(fires_shuffled) * 0.8)
    fires_train = fires_shuffled.iloc[:split_idx].copy()
    fires_test = fires_shuffled.iloc[split_idx:].copy()
    fires_train, fires_test = min_max_normalize_train_test(fires_train, fires_test, fires_cols_norm)

    train_rows = fires_train[fires_cols].to_dict('records')
    test_rows = fires_test[fires_cols].to_dict('records')
    y_train_fi = fires_train['area'].values
    y_test_fi = fires_test['area'].values

    null_pred_fi = regression_null_model(fires_train['area'])
    print("Null model MSE:", mean_squared_error([null_pred_fi] * len(y_test_fi), y_test_fi))

    D_fi = pairwise_fires(test_rows, train_rows, fires_cols, cyclical_cols, cycle_lengths, p=2)
    preds_fi = knn_regress(D_fi, y_train_fi, k=9, gamma=1.0)
    print("KNN (k=9, gamma=1.0) MSE:", mean_squared_error(preds_fi, y_test_fi))

    section("Tuning + 5x2cv demo (Breast Cancer)")

    demo_split_idx = int(len(cancer_shuffled) * 0.8)
    cancer_train_raw = cancer_shuffled.iloc[:demo_split_idx].copy()
    cancer_test_raw = cancer_shuffled.iloc[demo_split_idx:].copy()

    best_k, best_inner_err, all_k_results = tune_k_numeric_classification(
        cancer_train_raw, cancer_cols_norm, 'class',
        k_grid=[1, 3, 5, 7, 9, 11, 15], p=2, inner_k=5, random_state=RANDOM_STATE,
    )
    print("Inner-CV k search (k, mean error):", all_k_results)
    print("Best k:", best_k, "inner-CV error:", best_inner_err)

    n_cancer = len(cancer_shuffled)
    splits = five_by_two_split(n_cancer, random_state=RANDOM_STATE)
    knn_scores = np.zeros((5, 2))
    null_scores = np.zeros((5, 2))

    for rep, (fold_a, fold_b) in enumerate(splits):
        for fold_i, (train_idx, test_idx) in enumerate([fold_a, fold_b]):
            tr_df = cancer_shuffled.iloc[train_idx].copy()
            te_df = cancer_shuffled.iloc[test_idx].copy()
            tr_df, te_df = min_max_normalize_train_test(tr_df, te_df, cancer_cols_norm)

            null_pred = classification_null_model(tr_df['class'])
            null_scores[rep, fold_i] = classification_error(
                [null_pred] * len(te_df), te_df['class'].values)

            D = pairwise_numeric(te_df[cancer_cols_norm].values, tr_df[cancer_cols_norm].values, p=2)
            preds = knn_classify(D, tr_df['class'].values, best_k)
            knn_scores[rep, fold_i] = classification_error(preds, te_df['class'].values)

    print("Null model 5x2cv errors:\n", null_scores)
    print(f"KNN (k={best_k}) 5x2cv errors:\n{knn_scores}")

    t_stat, p_value, dof = dietterich_5x2cv_ttest(null_scores, knn_scores)
    print(f"Paired t-test (null vs KNN): t={t_stat:.4f}, p={p_value:.4f}, dof={dof}")
    if p_value < 0.05:
        print("--> KNN is significantly different from the null model at alpha=0.05.")
    else:
        print("--> No significant difference detected at alpha=0.05.")


    print("\nDONE -- all six data sets ran null model + KNN (+ Edited-KNN where shown),")
    print("plus a full tune -> 5x2cv -> significance-test demo on Breast Cancer.")
