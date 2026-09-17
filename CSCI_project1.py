import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from knn import Features, KNNClassifier, KNNRegressor, CHUNK_SIZE
from edited_condensed_knn import condensed_nn_classification, condensed_nn_regression, edited_nn_classification, edited_nn_regression
from Hyperparameter import tune_hyperparameters, classification_error, mean_squared_error


def makefeatures(df, numeric_cols, categorical_cols, cyclical_cols=None, cyc_periods=None):
    if cyclical_cols is None:
        cyclical_cols = []

    if cyc_periods is None:
        cyc_periods = []

    X_num = df[numeric_cols].to_numpy() if numeric_cols else np.empty((len(df), 0))

    X_cat = df[categorical_cols].to_numpy() if categorical_cols else np.empty((len(df), 0), dtype=object)

    X_cyc = df[cyclical_cols].to_numpy() if cyclical_cols else np.empty((len(df), 0))

    return Features(X_num = X_num, X_cat = X_cat, X_cyc = X_cyc, cyc_periods = cyc_periods)

def min_max_normalize(df, cols):
    df=df.copy()
    for col in cols:
       df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

def min_max_normalize_train_test(train_df, test_df, cols):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for col in cols:
        min_val = train_df[col].min()
        max_val = train_df[col].max()
        train_df[col] = (train_df[col] - min_val) / (max_val - min_val)
        test_df[col] = (test_df[col] - min_val) / (max_val - min_val)
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

def minkowski_distance(x, y, p):
    return np.sum(np.abs(x-y) ** p) **(1/p)

def compute_vdm_tables(train_df, cat_col,class_col):
    counts={}
    classes = train_df[class_col].unique()
    for col in train_df[cat_col].unique():
        subset = train_df[train_df[cat_col]==col]
        Ci = len(subset)
        Ci_a = {c: len(subset[subset[class_col]==c]) for c in classes}
        counts[col] = {'Ci': Ci, 'Ci_a': Ci_a}
    return counts

def vdm_delta(vi, vj, vdm_table, classes, p =1):
    if vi == vj:
        return 0.0

    Ci = vdm_table[vi]['Ci']
    Cj = vdm_table[vj]['Ci']
    total = 0.0

    for c in classes:
        Ci_a = vdm_table[vi]['Ci_a'][c]
        Cj_a = vdm_table[vj]['Ci_a'][c]

        total +=np.abs((Ci_a/Ci) - (Cj_a/Cj)) ** p
        
    return total

def cat_dist(x_cat, y_cat, vdm_tables, classes, p=1):
    dist = 0.0
    for k in range(len(x_cat)):
        dist +=vdm_delta(x_cat[k], y_cat[k], vdm_tables[k], classes, p=p)
    return dist ** (1/p)

def cyclical_distance(a, b, cycle_length):
    diff = abs(a - b)
    return min(diff, cycle_length - diff)

def fires_distance(x, y, cols, cyclical_cols, cyc_periods, p=2):
    total = 0.0
    for col in cols:
        if col in cyclical_cols:
            diff = cyclical_distance(x[col], y[col], cyc_periods[col])
        else:
            diff = abs(x[col] - y[col])
        total += diff ** p
    return total ** (1 / p)

def classification_error(y_pred, y_true):
    return (1/len(y_true)) * np.sum(y_pred !=y_true)

def mean_squared_error(y_pred, y_true):
    return (1/len(y_true)) * np.sum(y_pred - y_true) **2


#IMPORTANT: This is the Classification Null Model
def classification_null_model(y_train):
    counts = y_train.value_counts()
    tied_classes = counts[counts == counts.max()].index.tolist()
    return random.choice(tied_classes)

def regression_null_model(y_train):
    return y_train.mean()


def five_by_two_split(df, repeats=5, random_state=42):
    splits = []
    rng = np.random.default_rng(random_state)
    for _ in range(repeats):
        indices = rng.permutation(len(df))
        shuffled_df = df.iloc[indices].reset_index(drop=True)
        split = len(shuffled_df) // 2
        S1 = shuffled_df.iloc[:split]
        S2 = shuffled_df.iloc[split:]

        fold_A=(S1, S2)
        fold_B=(S2, S1)
        splits.append((fold_A, fold_B))
    return splits

def run_knn_classification_tuned(train_df, test_df, target_col, numeric_cols, categorical_cols, cyclical_cols=None, cyc_periods=None, k_values=[1, 3, 5], p_values=[1, 2],vdm_tables = None, classes = None):

    train_feats = makefeatures(train_df, numeric_cols, categorical_cols, cyclical_cols, cyc_periods)
    y_train = train_df[target_col].to_numpy()

    param_grid = {'k': list(k_values), 'p': list(p_values)}
    best_params, best_score, _ = tune_hyperparameters(knn_classifier_factory, train_feats, y_train, param_grid, metric = classification_error, lower_is_better=True, inner_k =5, random_state =0)

    test_feats = makefeatures(test_df, numeric_cols, categorical_cols, cyclical_cols, cyc_periods)
    final_model = KNNClassifier(k=best_params['k'], p=best_params['p'])
    final_model.fit(train_feats, y_train)
    y_pred = final_model.predict(test_feats)
    return y_pred, best_params

def run_knn_regression_tuned(train_df, test_df, target_col, numeric_cols, categorical_cols = None, cyclical_cols=None, cyc_periods=None, k_values=[1, 3, 5], p_values=[1, 2], gamma_grid = [0.1, 1.0, 10.0], vdm_tables = None, classes = None):

    train_feats = makefeatures(train_df, numeric_cols, categorical_cols, cyclical_cols, cyc_periods)
    y_train = train_df[target_col].to_numpy()

    param_grid = {'k': list(k_values), 'p': list(p_values), 'gamma': list(gamma_grid)}  # Include gamma in the parameter grid
    best_params, best_score, _ = tune_hyperparameters(knn_regressor_factory, train_feats, y_train, param_grid, metric = mean_squared_error, lower_is_better=True, inner_k =5, random_state =0)

    test_feats = makefeatures(test_df, numeric_cols, categorical_cols, cyclical_cols, cyc_periods)
    final_model = KNNRegressor(**best_params)
    final_model.fit(train_feats, y_train)
    y_pred = final_model.predict(test_feats)
    return y_pred, best_params

def run_edited_classification_tuned(train_df, test_df, target_col, numeric_cols, categorical_cols,
                                     cyclical_cols=None, cyc_periods=None,
                                     k_grid=(1, 3, 5, 7, 9), p_grid=(1, 2),
                                     vdm_tables=None, classes=None):
    train_feats = makefeatures(train_df, numeric_cols, categorical_cols, cyclical_cols, cyc_periods)
    y_train = train_df[target_col].to_numpy()

    param_grid = {'k': list(k_grid), 'p': list(p_grid)}
    best_params, best_score, _ = tune_hyperparameters(
        edited_classifier_factory, train_feats, y_train, param_grid,
        metric=classification_error, lower_is_better=True, inner_k=5, random_state=0
    )

    # rerun editing on the FULL outer train set using the best p, then fit final k-NN
    reduced_feats, reduced_y, _ = edited_nn_classification(train_feats, y_train, p=best_params['p'])
    test_feats = makefeatures(test_df, numeric_cols, categorical_cols, cyclical_cols, cyc_periods)
    final_model = KNNClassifier(k=best_params['k'], p=best_params['p'])
    final_model.fit(reduced_feats, reduced_y)
    predictions = final_model.predict(test_feats)
    return predictions, best_params
    

def run_condensed_regression_tuned(train_df, test_df, target_col, numeric_cols, categorical_cols=None,
                                    cyclical_cols=None, cyc_periods=None,
                                    k_grid=(1, 3, 5, 7, 9), p_grid=(2,), gamma_grid=(0.1, 1, 10),
                                    epsilon_grid=(10, 50, 100),
                                    vdm_tables=None, classes=None):
    train_feats = makefeatures(train_df, numeric_cols, categorical_cols, cyclical_cols, cyc_periods)
    y_train = train_df[target_col].to_numpy(dtype=float)

    param_grid = {'k': list(k_grid), 'p': list(p_grid), 'gamma': list(gamma_grid), 'epsilon': list(epsilon_grid)}
    best_params, best_score, _ = tune_hyperparameters(
        condensed_regressor_factory, train_feats, y_train, param_grid,
        metric=mean_squared_error, lower_is_better=True, inner_k=5, random_state=0
    )

    # rerun condensing on the FULL outer train set using the best epsilon/p/gamma, then fit final k-NN
    reduced_feats, reduced_y, _ = condensed_nn_regression(
        train_feats, y_train, epsilon=best_params['epsilon'], p=best_params['p'], gamma=best_params['gamma']
    )
    test_feats = makefeatures(test_df, numeric_cols, categorical_cols, cyclical_cols, cyc_periods)
    final_model = KNNRegressor(k=best_params['k'], p=best_params['p'], gamma=best_params['gamma'])
    final_model.fit(reduced_feats, reduced_y)
    predictions = final_model.predict(test_feats)
    return predictions, best_params

def run_condensed_classification_tuned(train_df, test_df, target_col, numeric_cols, categorical_cols,
                                        cyclical_cols=None, cyc_periods=None,
                                        k_grid=(1, 3, 5, 7, 9), p_grid=(1, 2),
                                        vdm_tables=None, classes=None):
    train_feats = makefeatures(train_df, numeric_cols, categorical_cols, cyclical_cols, cyc_periods)
    y_train = train_df[target_col].to_numpy()

    param_grid = {'k': list(k_grid), 'p': list(p_grid)}
    best_params, best_score, _ = tune_hyperparameters(
        condensed_classifier_factory, train_feats, y_train, param_grid,
        metric=classification_error, lower_is_better=True, inner_k=5, random_state=0
    )

    # rerun condensing on the FULL outer train set using the best p, then fit final k-NN
    reduced_feats, reduced_y, _ = condensed_nn_classification(train_feats, y_train, p=best_params['p'])
    test_feats = makefeatures(test_df, numeric_cols, categorical_cols, cyclical_cols, cyc_periods)
    final_model = KNNClassifier(k=best_params['k'], p=best_params['p'])
    final_model.fit(reduced_feats, reduced_y)
    predictions = final_model.predict(test_feats)
    return predictions, best_params

def run_edited_regression_tuned(train_df, test_df, target_col, numeric_cols, categorical_cols=None,
                                 cyclical_cols=None, cyc_periods=None,
                                 k_grid=(1, 3, 5, 7, 9), p_grid=(2,), gamma_grid=(0.1, 1, 10),
                                 epsilon_grid=(10, 50, 100),
                                 vdm_tables=None, classes=None):
    train_feats = makefeatures(train_df, numeric_cols, categorical_cols, cyclical_cols, cyc_periods)
    y_train = train_df[target_col].to_numpy(dtype=float)

    param_grid = {'k': list(k_grid), 'p': list(p_grid), 'gamma': list(gamma_grid), 'epsilon': list(epsilon_grid)}
    best_params, best_score, _ = tune_hyperparameters(
        edited_regressor_factory, train_feats, y_train, param_grid,
        metric=mean_squared_error, lower_is_better=True, inner_k=5, random_state=0
    )

    # rerun editing on the FULL outer train set using the best epsilon/p/gamma, then fit final k-NN
    reduced_feats, reduced_y, _ = edited_nn_regression(
        train_feats, y_train, epsilon=best_params['epsilon'], p=best_params['p'], gamma=best_params['gamma']
    )
    test_feats = makefeatures(test_df, numeric_cols, categorical_cols, cyclical_cols, cyc_periods)
    final_model = KNNRegressor(k=best_params['k'], p=best_params['p'], gamma=best_params['gamma'])
    final_model.fit(reduced_feats, reduced_y)
    predictions = final_model.predict(test_feats)
    return predictions, best_params


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

def edited_regressor_factory(k, p, gamma, epsilon):
    def fit_predict(train_feats, train_y, val_feats):
        reduced_feats, reduced_y, _ = edited_nn_regression(train_feats, train_y,epsilon=epsilon, p=p, gamma=gamma)
        model = KNNRegressor(k=5, p=p, gamma=gamma)
        model.fit(reduced_feats, reduced_y)
        return model.predict(val_feats)
    return fit_predict

def condensed_classifier_factory(k,p):
    def fit_predict(train_feats, train_y, val_feats):
        reduced_feats, reduced_y, _ = condensed_nn_classification(train_feats, train_y, p=p)
        model = KNNClassifier(k=k,p=p)
        model.fit(reduced_feats, reduced_y)
        return model.predict(val_feats)
    return fit_predict

def edited_classifier_factory(k, p):
    def fit_predict(train_feats, train_y, val_feats):
        reduced_feats, reduced_y, _ = edited_nn_classification(train_feats, train_y, p=p)
        model = KNNClassifier(k=k, p=p)
        model.fit(reduced_feats, reduced_y)
        return model.predict(val_feats)
    return fit_predict

def condensed_regressor_factory(k, p, gamma, epsilon):
    def fit_predict(train_feats, train_y, val_feats):
        reduced_feats, reduced_y, _ = condensed_nn_regression(train_feats, train_y, epsilon=epsilon, p=p, gamma=gamma)
        model = KNNRegressor(k=k, p=p, gamma=gamma)
        model.fit(reduced_feats, reduced_y)
        return model.predict(val_feats)
    return fit_predict


def run_cv_for_dataset(df, target_col, numeric_cols, categorical_cols, methods, is_classification, repeats=5, cyclical_cols=None, cyc_periods=None):
    splits = five_by_two_split(df, repeats=5, random_state=42)
    results = {name: [] for name in methods}
    best_params_results = {name: [] for name in methods}

    for fold_A, fold_B in splits:
        for train_df, test_df in [fold_A, fold_B]:

            vdm_tables = None
            classes = None

            if categorical_cols:
                # Bug 2 fix: use 'col', not 'categorical_cols', as the cat_col argument
                vdm_tables = [compute_vdm_tables(train_df, col, target_col) for col in categorical_cols]
                classes = train_df[target_col].unique()
            else:
                # Bug 4 fix: use the train/test-aware normalizer, and actually keep the result
                train_df, test_df = min_max_normalize_train_test(train_df, test_df, numeric_cols)

            y_test = test_df[target_col]

            for name, method_fn in methods.items():

                result = method_fn(
                    train_df,
                    test_df,
                    target_col,
                    numeric_cols,
                    categorical_cols
                )

                # Tuned methods return (predictions, best_params)
                if isinstance(result, tuple):
                    y_pred, best_params = result
                    best_params_results[name].append(best_params)
                else:
                    y_pred = result

                if is_classification:
                    score = classification_error(y_test, y_pred)
                else:
                    score = mean_squared_error(y_test, y_pred)

                results[name].append(score)

    return results, best_params_results

def run_null_regression(train_df, test_df, target_col, vdm_tables=None, classes=None):
    pred_value = regression_null_model(train_df[target_col])
    return [pred_value] * len(test_df)

def run_null_classification(train_df, test_df, target_col, vdm_tables=None, classes=None):
    pred_value = classification_null_model(train_df[target_col])
    return [pred_value] * len(test_df)

def run_knn_regression(train_df, test_df, target_col, numeric_cols, categorical_cols, cyclical_cols=None, cyc_periods=None, k = 5, gamma=1.0, p=2):
    train_feats = makefeatures(train_df, numeric_cols, categorical_cols, cyclical_cols, cyc_periods)
    test_feats = makefeatures(test_df, numeric_cols, categorical_cols, cyclical_cols, cyc_periods)
    model = KNNRegressor(k=k, p=p, gamma=gamma)
    model.fit(train_feats, train_df[target_col].to_numpy())
    return model.predict(test_feats)

def run_knn_classification(train_df, test_df, target_col, numeric_cols, categorical_cols, cyclical_cols=None, cyc_periods=None, k=5, p=2, random_state=None):
    train_feats = makefeatures(train_df, numeric_cols, categorical_cols, cyclical_cols, cyc_periods)
    test_feats = makefeatures(test_df, numeric_cols, categorical_cols, cyclical_cols, cyc_periods)
    model = KNNClassifier(k=k, p=p, random_state=random_state)
    model.fit(train_feats, train_df[target_col].to_numpy())
    return model.predict(test_feats)

def run_edited_regression(train_df, test_df, target_col, numeric_cols, categorical_cols, epsilon = 0.1, k=5, p=2, max_iters=50, random_state=None):
    train_feats = makefeatures(train_df, numeric_cols, categorical_cols, [], [])
    test_feats = makefeatures(test_df, numeric_cols, categorical_cols, [], [])
    y_train = train_df[target_col].to_numpy()

    reduced_feats, reduced_y, keep = edited_nn_regression(train_feats, y_train, epsilon=epsilon, p=p, max_iters=max_iters, random_state=random_state)

    knn=KNNRegressor(k=k, p=p, random_state=random_state)
    knn.fit(reduced_feats, reduced_y)

    predictions = knn.predict(test_feats)
    return predictions

def run_edited_classification(train_df, test_df, target_col, numeric_cols, categorical_cols, epsilon=0.1, k=5, p=2, max_iters=50, random_state=None):
    train_feats = makefeatures(train_df, numeric_cols, categorical_cols, [], [])
    test_feats = makefeatures(test_df, numeric_cols, categorical_cols, [], [])
    y_train = train_df[target_col].to_numpy()

    reduced_feats, reduced_y, keep = edited_nn_classification(train_feats, y_train, p=p, max_iters=max_iters, random_state=random_state)

    knn=KNNClassifier(k=k, p=p, random_state=random_state)
    knn.fit(reduced_feats, reduced_y)

    predictions = knn.predict(test_feats)
    return predictions

def run_condensed_regression(train_df, test_df, target_col, numeric_cols, categorical_cols, epsilon=0.1, k=5, p=2, gamma=1.0, max_iters=50, random_state=None):
    train_feats = makefeatures(train_df, numeric_cols, categorical_cols, [], [])
    test_feats = makefeatures(test_df, numeric_cols, categorical_cols, [], [])
    y_train = train_df[target_col].to_numpy()

    reduced_feats, reduced_y, keep = condensed_nn_regression(train_feats, y_train, epsilon=epsilon, p=p, max_iters=max_iters, random_state=random_state)

    knn=KNNRegressor(k=k, p=p, gamma=gamma, random_state=random_state)
    knn.fit(reduced_feats, reduced_y)

    predictions = knn.predict(test_feats)
    return predictions

def run_condensed_classification(train_df, test_df, target_col, numeric_cols, categorical_cols, k=5, p=2, max_iters=50, random_state=None):
    train_feats = makefeatures(train_df, numeric_cols, categorical_cols, [], [])
    test_feats = makefeatures(test_df, numeric_cols, categorical_cols, [], [])
    y_train = train_df[target_col].to_numpy()

    reduced_feats, reduced_y, keep = condensed_nn_classification(train_feats, y_train, p=p, max_iters=max_iters, random_state=random_state)
    print("Condensed size:", len(reduced_y), " Classes present:", set(reduced_y))  # add this

    knn=KNNClassifier(k=k, p=p, random_state=random_state)
    knn.fit(reduced_feats, reduced_y)

    predictions = knn.predict(test_feats)
    return predictions


def knn_regress_predict(train_df, query_row, feature_cols, target_col, k, distance_fn, p=2):
    distances = []
    for _, train_row in train_df.iterrows():
        x = np.array([train_row[c] for c in feature_cols])
        y = np.array([query_row[c] for c in feature_cols])
        d = distance_fn(x, y, p)
        distances.append((d, train_row[target_col]))
    distances.sort(key=lambda pair: pair[0])
    nearest = distances[:k]
    weights = [np.exp(-gamma * d**2) for d, _ in nearest]
    values = [v for _, v in nearest]
    if sum(weights) == 0:
        return np.mean(values)
    return sum(w * v for w, v in zip(weights, values)) / sum(weights)



machine_cols = ['vendor_name', 'model_name', 'myct', 'mmin', 'mmax', 'cach', 'chmin', 'chmax', 'prp', 'erp']
machine_numeric_cols = ['myct', 'mmin', 'mmax', 'cach', 'chmin', 'chmax']
machine = pd.read_csv("machine.data", header=None, names=machine_cols)
erp = machine['erp']
machine = machine.drop(columns = ['vendor_name', 'model_name', 'erp'])
#print("machine data set")


abalone_cols= ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings']
abalone = pd.read_csv("abalone.data", header=None, names=abalone_cols)
abalone = pd.get_dummies(abalone, columns=['sex'], dtype=int)
print("abalone data set")
print(abalone.head(10))
abalone_cols_norm = ['length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']
abalone_numeric_cols = ['length', 'diameter', 'height', 'whole_weight',
                         'shucked_weight', 'viscera_weight', 'shell_weight'] + \
                        [c for c in abalone.columns if c.startswith('sex_')]
min_max_normalized_abalone = min_max_normalize(abalone, abalone_cols_norm)
#print(min_max_normalized_abalone.head(10))



car_cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
car_categorical_cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']


car = pd.read_csv("car.data", header=None, names=car_cols)
#car_vdm_tables = [compute_vdm_tables(car, col, 'class') for col in car_cat_cols]
classes = car['class'].unique()
row1 = car.iloc[0]
row2 = car.iloc[1]
#x_cat = [row1[col] for col in car_cat_cols]
#y_cat = [row2[col] for col in car_cat_cols]
#distance = cat_dist(x_cat, y_cat, car_vdm_tables, classes, p=1)
#print(distance)
#print(cat_dist(x_cat, x_cat, car_vdm_tables, classes, p=1))
#y_train = car.iloc[:int(len(car)*0.8)]['class']
#y_test = car.iloc[int(len(car)*0.8):]['class']

#pred = classification_null_model(y_train)
#y_pred = [pred] * len(y_test)
#error = classification_error(y_test, y_pred)
#print("Car null model error:", error)

cancer_cols = ['id', 'clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity',
               'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
               'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
cancer = pd.read_csv("breast-cancer-wisconsin.data", header = None, names = cancer_cols)
cancer = cancer.drop(columns=['id'])
cancer = cancer[cancer['bare_nuclei'] != '?']
cancer['bare_nuclei'] = cancer['bare_nuclei'].astype(int)

#print(cancer.head(10))
cancer_numeric_cols = ['clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity',
                        'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
                        'bland_chromatin', 'normal_nucleoli', 'mitoses']
#print(min_max_normalized_cancer.head(10))


month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
day_order = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

fires = pd.read_csv("forestfires.csv")
fires['area'] = np.log1p(fires['area'])
fires['month'] = fires['month'].apply(lambda x: month_order.index(x))
fires['day'] = fires['day'].apply(lambda x: day_order.index(x))
#print("fires data set")
#print(fires.head(10))
fires_cols_norm = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
min_max_normalized_fires = min_max_normalize(fires, fires_cols_norm)
#print(min_max_normalized_fires.head(10))
fires_cols = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
cyclical_cols = ['month', 'day']
cycle_lengths = {'month': 12, 'day': 7}
row1 = fires.iloc[0]
row2 = fires.iloc[1]
dist = fires_distance(row1, row2, fires_cols, cyclical_cols, cycle_lengths, p=2)
print(dist)
fires_numeric_cols = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']


    
house_data = load_vote("house-votes-84.data")
house_df = pd.DataFrame(house_data, columns=['party', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cut', 'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa'])
house_categorical_cols = ['handicapped-infants', 'water-project-cost-sharing',
                           'adoption-of-the-budget-resolution', 'physician-fee-freeze',
                           'el-salvador-aid', 'religious-groups-in-schools',
                           'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
                           'mx-missile', 'immigration', 'synfuels-corporation-cut',
                           'education-spending', 'superfund-right-to-sue', 'crime',
                           'duty-free-exports', 'export-administration-act-south-africa']
#print(house_df.head(10))





methods_regression = {
    'null regression': run_null_regression,
    'knn regression': lambda train, test, target, numeric_cols, categorical_cols, cyclical_cols=None, cycle_lengths=None: run_knn_regression(
        train, test, target, numeric_cols = numeric_cols, categorical_cols = categorical_cols, k=5, gamma=1.0, p=2
    ),
    'knn regression (tuned)': lambda train, test, target, numeric_cols, categorical_cols, **kw: run_knn_regression_tuned(
            train, test, target, numeric_cols, categorical_cols,
            k_values=(3, 5, 7), p_values=(1,2), gamma_grid=(1.0,)  # start small, expand later
        ),
    'edited regression': lambda train, test, target, numeric_cols, categorical_cols, cyclical_cols=None, cycle_lengths=None, random_state=None: run_edited_regression(
        train, test, target, numeric_cols = numeric_cols, categorical_cols = categorical_cols, epsilon=50, k=5, p=2, max_iters=50, random_state=random_state
    ),
    'edited regression (tuned)': lambda train, test, target, numeric_cols, categorical_cols, **kw: run_edited_regression_tuned(
        train, test, target, numeric_cols, categorical_cols,
        k_grid=(3, 5, 7), gamma_grid=(1.0,), epsilon_grid=(25, 50, 100)   # start small, expand later
    ),
    'condensed regression': lambda train, test, target, numeric_cols, categorical_cols, cyclical_cols=None, cycle_lengths=None, random_state=None: run_condensed_regression(
        train, test, target, numeric_cols = numeric_cols, categorical_cols = categorical_cols, epsilon=50, k=5, p=2, gamma=1.0, max_iters=50, random_state=random_state
    ),
    'condensed regression (tuned)': lambda train, test, target, numeric_cols, categorical_cols, **kw: run_condensed_regression_tuned(
    train, test, target,
    numeric_cols,
    categorical_cols,
    k_grid=(3, 5, 7),
    p_grid=(1, 2),
    gamma_grid=(1.0,),
    epsilon_grid=(25, 50, 100)
),

}
methods_classification = {
    'null classification': run_null_classification,
    'knn classification': lambda train, test, target, numeric_cols, categorical_cols, cyclical_cols=None, cycle_lengths=None: run_knn_classification(
        train, test, target, numeric_cols=numeric_cols, categorical_cols=categorical_cols, k=5, p=1
    ),
    
    'edited classification': lambda train, test, target, numeric_cols, categorical_cols, cyclical_cols=None, cycle_lengths=None, random_state=None: run_edited_classification(
        train, test, target, numeric_cols = numeric_cols, categorical_cols = categorical_cols, epsilon=0.1, k=5, p=2, max_iters=50, random_state=random_state
    ),
    'edited classification (tuned)': lambda train, test, target, numeric_cols, categorical_cols, cyclical_cols=None, cycle_lengths=None, random_state=None: run_edited_classification_tuned(
    train, test, target,
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    k_grid=(1, 3, 5, 7, 9),
    p_grid=(1, 2)
),
    'condensed classification': lambda train, test, target, numeric_cols, categorical_cols, cyclical_cols=None, cycle_lengths=None, random_state=None: run_condensed_classification(
        train, test, target, numeric_cols = numeric_cols, categorical_cols = categorical_cols, k=5, p=2, max_iters=50, random_state=42
    ),
    'condensed classification (tuned)': lambda train, test, target, numeric_cols, categorical_cols, cyclical_cols=None, cycle_lengths=None: run_condensed_classification_tuned(
        train, test, target, numeric_cols, categorical_cols,
            k_grid=(3, 5, 7), p_grid=(1, 2)   # start small, expand later
        ),
}

classification_hyperparams = {
    'null classification': 'None',

    'knn classification': {
        'k': 5,
        'p': 1
    },

    'edited classification': {
        'epsilon': 0.1,
        'k': 5,
        'p': 2,
        'max_iters': 50
    },

    'condensed classification': {
        'k': 5,
        'p': 2,
        'max_iters': 50,
        'random_state': 42
    },

    'condensed classification (tuned)': {
        'k_grid': (3, 5, 7),
        'p_grid': (1, 2)
    }
}


regression_hyperparams = {
    'null regression': 'None',

    'knn regression': {
        'k': 5,
        'gamma': 1.0,
        'p': 2
    },

    'knn regression (tuned)': {
        'k_values': (3, 5, 7),
        'p_values': (1, 2),
        'gamma_grid': (1.0,)
    },

    'edited regression': {
        'epsilon': 50,
        'k': 5,
        'p': 2,
        'max_iters': 50
    },

    'edited regression (tuned)': {
        'k_grid': (3, 5, 7),
        'gamma_grid': (1.0,),
        'epsilon_grid': (25, 50, 100)
    },

    'condensed regression': {
        'epsilon': 50,
        'k': 5,
        'p': 2,
        'gamma': 1.0,
        'max_iters': 50
    }
}


"""results_regression = run_cv_for_dataset(machine, 'prp', machine_numeric_cols, [], methods_regression,
                              is_classification=False)
print(results_regression)
print("Null model average MSE:", np.mean(results_regression['null regression']))
print("k-NN average MSE:", np.mean(results_regression['knn regression']))
results_classification = run_cv_for_dataset(car, 'class', [], car_categorical_cols, methods_classification,
                              is_classification=True)
print(results_classification)"""

def run_and_report(df, target_col, numeric_cols, categorical_cols, methods, is_classification, name,
                   cyclical_cols=None, cycle_lengths=None):

    print(f"\n{'='*70}\nRunning {name}\n{'='*70}")
    t0 = time.time()

    if cyclical_cols:
        results, best_params_results = run_cv_for_dataset(
            df, target_col, numeric_cols, categorical_cols, methods,
            is_classification=is_classification,
            cyclical_cols=cyclical_cols,
            cyc_periods=cycle_lengths
        )
    else:
        results, best_params_results = run_cv_for_dataset(
            df, target_col, numeric_cols, categorical_cols, methods,
            is_classification=is_classification
        )

    # Select the appropriate hyperparameter dictionary
    hyperparams = (
        classification_hyperparams
        if is_classification
        else regression_hyperparams
    )

    print(f"\n{name} results:")

    for method_name, scores in results.items():

        print(f"\n  {method_name}:")

        # Print best parameters selected during each outer fold
        if best_params_results[method_name]:
            for fold_num, params in enumerate(best_params_results[method_name], start=1):
                print(f"      Fold {fold_num}: ", end="")

                if isinstance(params, dict):
                    print(", ".join(f"{key}={value}" for key, value in params.items()))
                else:
                    print(params)

        else:
            print("      No tuned hyperparameters")

        print(f"      avg = {np.mean(scores):.4f}   std = {np.std(scores):.4f}")

    print(f"\n{name} took {time.time()-t0:.1f} seconds")

    return results


# ==================== Regression datasets ====================

results_machine = run_and_report(
    machine, 'prp', machine_numeric_cols, [], methods_regression,
    is_classification=False, name="Computer Hardware"
)

results_abalone = run_and_report(
    abalone, 'rings', abalone_numeric_cols, [], methods_regression,
    is_classification=False, name="Abalone"
)

results_fires = run_and_report(
    fires, 'area', fires_numeric_cols, [], methods_regression,
    is_classification=False, name="Forest Fires",
    cyclical_cols=cyclical_cols, cycle_lengths=cycle_lengths
)


# ==================== Classification datasets ====================

results_car = run_and_report(
    car, 'class', [], car_categorical_cols, methods_classification,
    is_classification=True, name="Car Evaluation"
)

results_house = run_and_report(
    house_df, 'party', [], house_categorical_cols, methods_classification,
    is_classification=True, name="Congressional Vote"
)

results_cancer = run_and_report(
    cancer, 'class', cancer_numeric_cols, [], methods_classification,
    is_classification=True, name="Breast Cancer"
)


# ==================== Final summary ====================

print(f"\n{'='*70}\nALL DATASETS COMPLETE\n{'='*70}")
all_results = {
    'Computer Hardware': results_machine,
    'Abalone': results_abalone,
    'Forest Fires': results_fires,
    'Car Evaluation': results_car,
    'Congressional Vote': results_house,
    'Breast Cancer': results_cancer,
}
for dataset_name, res in all_results.items():
    print(f"\n{dataset_name}:")
    for method_name, scores in res.items():
        print(f"  {method_name:35s} avg = {np.mean(scores):.4f}")

#train_norm, test_norm = min_max_normalize_train_test(train_df, test_df, machine_numeric_cols)

#print("Original training set size:", len(train_norm))
#edited_train = edit_dataset_regression(train_norm, machine_numeric_cols, 'prp',
     #                                   minkowski_distance, epsilon=50.0, p=2, gamma=1.0)
#print("Edited training set size:", len(edited_train))

"""y_pred_edited = run_knn_regression(edited_train, test_norm, 'prp', machine_numeric_cols, k=5, gamma=1.0)
y_pred_plain = run_knn_regression(train_norm, test_norm, 'prp', machine_numeric_cols, k=5, gamma=1.0)
y_true = test_norm['prp']

print("Edited k-NN MSE:", mean_squared_error(y_true, y_pred_edited))
print("Plain k-NN MSE:", mean_squared_error(y_true, y_pred_plain))"""

