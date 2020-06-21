import pickle
import numpy as np
import pandas as pd


dz_list = [5, 10, 15, 20]
cat_list = ['A', 'B', 'C', 'D']
num_th = 20
df = pd.DataFrame()
for run in range(1, 11):
    est_cmi = []
    true_cmi = []
    for cat, dz in zip(cat_list, dz_list):
        file_name = f'./run{run}/cat{cat}.{num_th}k.dz{dz}.seed0.est_cmi.migan.txt'
        with open(file_name, 'rb') as fp:
            cmi_buf = pickle.load(fp)
        est_cmi.append(np.mean(cmi_buf[2500:]))

    df[f'Run {run}'] = est_cmi

for cat, dz in zip(cat_list, dz_list):
    true_cmi.append(np.load(f'../data/cat{cat}/ksg_gt.dz{dz}.npy').astype(np.float32)[0])

df = df.clip(lower=0.0)
mean_est = df.mean(axis=1)
est_var = df.var(axis=1)
df['Avg Est'] = mean_est
df['Var'] = est_var
df['True CMI'] = true_cmi
df.to_csv('./est_cmi.csv')
