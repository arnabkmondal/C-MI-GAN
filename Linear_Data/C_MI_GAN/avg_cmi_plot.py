import pickle
import numpy as np
import pandas as pd

dz = [1, 10, 50, 100]
df = pd.DataFrame(
    {'data': ['F_20_5', 'F_20_10', 'F_20_20', 'F_20_50', 'F_1_20', 'F_10_20', 'F_50_20', 'F_100_20',
              'G_20_5', 'G_20_10', 'G_20_20', 'G_20_50', 'G_1_20', 'G_10_20', 'G_50_20', 'G_100_20']})
for run in range(1, 11):
    est_cmi = []
    true_cmi = []
    for cat in ['F', 'G']:
        for i in [5, 10, 20, 50]:
            file_name = f'./run{run}/cat{cat}.{i}k.dz20.seed0.est_cmi.migan.txt'
            with open(file_name, 'rb') as fp:
                cmi_buf = pickle.load(fp)
            est_cmi.append(np.mean(cmi_buf[2000:]))
            true_cmi.append(np.load(f'../data/cat{cat}/ksg_gt.dz20.npy')[0])

        for i in dz:
            file_name = f'./run{run}/cat{cat}.20k.dz{i}.seed0.est_cmi.migan.txt'
            with open(file_name, 'rb') as fp:
                cmi_buf = pickle.load(fp)

            est_cmi.append(np.mean(cmi_buf[2000:]))
            true_cmi.append(np.load(f'../data/cat{cat}/ksg_gt.dz{i}.npy')[0])

    df[f'Run {run}'] = est_cmi
df = df.clip(lower=0.0)
mean_est = df.mean(axis=1)
est_var = df.var(axis=1)
df['Avg Est'] = mean_est
df['Var'] = est_var
df['True CMI'] = true_cmi[0:16]
df.to_csv('./est_cmi.csv')
