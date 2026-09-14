import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from knn import Features, KNNClassifier, KNNRegressor, CHUNK_SIZE
from edited_condensed_knn import condensed_nn_classification, condensed_nn_regression, edited_nn_classification, edited_nn_regression
 


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



def run_cv_for_dataset(df, target_col, numeric_cols, categorical_cols, methods, is_classification, repeats=5, cyclical_cols=None, cyc_periods=None):
    splits = five_by_two_split(df, repeats=5, random_state=42)
    results = {name: [] for name in methods}

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
                # Bug 3 fix: pass vdm_tables/classes through so cat_dist-based methods can use them
                y_pred = method_fn(train_df, test_df, target_col, numeric_cols, categorical_cols)

                if is_classification:
                    score = classification_error(y_test, y_pred)
                else:
                    score = mean_squared_error(y_test, y_pred)

                results[name].append(score)

    return results

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
abalone = pd.get_dummies(abalone, columns=['sex'])
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
    'edited regression': lambda train, test, target, numeric_cols, categorical_cols, cyclical_cols=None, cycle_lengths=None, random_state=None: run_edited_regression(
        train, test, target, numeric_cols = numeric_cols, categorical_cols = categorical_cols, epsilon=0.1, k=5, p=2, max_iters=50, random_state=random_state
    ),
    'condensed regression': lambda train, test, target, numeric_cols, categorical_cols, cyclical_cols=None, cycle_lengths=None, random_state=None: run_condensed_regression(
        train, test, target, numeric_cols = numeric_cols, categorical_cols = categorical_cols, k=5, p=2, gamma=1.0, max_iters=50, random_state=random_state
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
    'condensed classification': lambda train, test, target, numeric_cols, categorical_cols, cyclical_cols=None, cycle_lengths=None, random_state=None: run_condensed_classification(
        train, test, target, numeric_cols = numeric_cols, categorical_cols = categorical_cols, k=5, p=2, max_iters=50, random_state=random_state
    )
}


results_regression = run_cv_for_dataset(machine, 'prp', machine_numeric_cols, [], methods_regression,
                              is_classification=False)
print(results_regression)
print("Null model average MSE:", np.mean(results_regression['null regression']))
print("k-NN average MSE:", np.mean(results_regression['knn regression']))
results_classification = run_cv_for_dataset(car, 'class', [], car_categorical_cols, methods_classification,
                              is_classification=True)
print(results_classification)

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


