import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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

def fires_distance(x, y, cols, cyclical_cols, cycle_lengths, p=2):
    total = 0.0
    for col in cols:
        if col in cyclical_cols:
            diff = cyclical_distance(x[col], y[col], cycle_lengths[col])
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

machine_cols = ['vendor_name', 'model_name', 'myct', 'mmin', 'mmax', 'cach', 'chmin', 'chmax', 'prp', 'erp']
machine = pd.read_csv("machine.data", header=None, names=machine_cols)
erp = machine['erp']
machine = machine.drop(columns = ['vendor_name', 'model_name', 'erp'])
print("machine data set")
print(machine.head(10))
machine_cols_norm = ['myct', 'mmin', 'mmax', 'cach', 'chmin', 'chmax']
min_max_normalized_machine = min_max_normalize(machine, machine_cols_norm)
print(min_max_normalized_machine.head(10))
print("\n")
machine_shuffled = machine.sample(frac=1, random_state=42).reset_index(drop=True)

split_idx = int(len(machine_shuffled) * 0.8)
y_train = machine_shuffled.iloc[:split_idx]['prp']
y_test = machine_shuffled.iloc[split_idx:]['prp']

pred_value = regression_null_model(y_train)
y_pred = [pred_value] * len(y_test)
error = mean_squared_error(y_test, y_pred)

print("Predicted value:", pred_value)
print("Machine null model MSE:", error)

print("\n")
abalone_cols= ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings']
abalone = pd.read_csv("abalone.data", header=None, names=abalone_cols)
abalone = pd.get_dummies(abalone, columns=['sex'])
print("abalone data set")
print(abalone.head(10))
abalone_cols_norm = ['length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']
min_max_normalized_abalone = min_max_normalize(abalone, abalone_cols_norm)
print(min_max_normalized_abalone.head(10))
print("\n")



car_cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
car_cat_cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

car = pd.read_csv("car.data", header=None, names=car_cols)
car_vdm_tables = [compute_vdm_tables(car, col, 'class') for col in car_cat_cols]
classes = car['class'].unique()
row1 = car.iloc[0]
row2 = car.iloc[1]
x_cat = [row1[col] for col in car_cat_cols]
y_cat = [row2[col] for col in car_cat_cols]
distance = cat_dist(x_cat, y_cat, car_vdm_tables, classes, p=1)
print(distance)
print(cat_dist(x_cat, x_cat, car_vdm_tables, classes, p=1))
print("car data set")
print(car.head(10))
print("\n")

y_train = car.iloc[:int(len(car)*0.8)]['class']
y_test = car.iloc[int(len(car)*0.8):]['class']

pred = classification_null_model(y_train)
y_pred = [pred] * len(y_test)
error = classification_error(y_test, y_pred)
print("Car null model error:", error)
print("\n")

cancer_cols = ['id', 'clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity',
               'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
               'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
cancer = pd.read_csv("breast-cancer-wisconsin.data", header = None, names = cancer_cols)
cancer = cancer.drop(columns=['id'])
cancer = cancer[cancer['bare_nuclei'] != '?']
cancer['bare_nuclei'] = cancer['bare_nuclei'].astype(int)
print("cancer data set")
print(cancer.head(10))
cancer_cols_norm = ['clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity',
                    'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
                    'bland_chromatin', 'normal_nucleoli', 'mitoses']
min_max_normalized_cancer = min_max_normalize(cancer, cancer_cols_norm)
print(min_max_normalized_cancer.head(10))
print("\n")

month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
day_order = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

fires = pd.read_csv("forestfires.csv")
fires['area'] = np.log1p(fires['area'])
fires['month'] = fires['month'].apply(lambda x: month_order.index(x))
fires['day'] = fires['day'].apply(lambda x: day_order.index(x))
print("fires data set")
print(fires.head(10))
fires_cols_norm = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
min_max_normalized_fires = min_max_normalize(fires, fires_cols_norm)
print(min_max_normalized_fires.head(10))
fires_cols = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
cyclical_cols = ['month', 'day']
cycle_lengths = {'month': 12, 'day': 7}
row1 = fires.iloc[0]
row2 = fires.iloc[1]
dist = fires_distance(row1, row2, fires_cols, cyclical_cols, cycle_lengths, p=2)
print(dist)
print("\n")
    
house_data = load_vote("house-votes-84.data")
house_df = pd.DataFrame(house_data, columns=['party', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cut', 'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa'])
print("house data set")
print(house_df.head(10))

