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
car = pd.read_csv("car.data", header=None, names=car_cols)
print("car data set")
print(car.head(10))
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

fires = pd.read_csv("forestfires.csv")
fires['area'] = np.log1p(fires['area'])
print("fires data set")
print(fires.head(10))
fires_cols_norm = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
min_max_normalized_fires = min_max_normalize(fires, fires_cols_norm)
print(min_max_normalized_fires.head(10))
print("\n")
    
house_data = load_vote("house-votes-84.data")
house_df = pd.DataFrame(house_data, columns=['party', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cut', 'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa'])
print("house data set")
print(house_df.head(10))

