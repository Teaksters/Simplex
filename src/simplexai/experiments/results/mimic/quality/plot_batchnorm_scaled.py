import argparse
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import torch
import os
import io


class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-cv_list",
    nargs="+",
    default=[0, 1, 2, 3, 4, 5],
    help="The list of experiment cv identifiers to plot",
    type=int,
)
parser.add_argument(
    "-age_scalers",
    nargs='*',
    type=float,
    default=[1.0, 2.0, 3.0, 4.0, 5.0],
    help="Scaling variable for sample ages")

args = parser.parse_args()
cv_list = args.cv_list
scalers = args.age_scalers

current_path = Path.cwd()
load_path = current_path / "experiments/results/mimic/quality/scaled/"

# Gather all data based on scaler
all_data = []
for scaler in scalers:
    scaler_data = []
    for cv in cv_list:
        temp_path = load_path / str(scaler) / f"BNorm_out_data_cv{cv}.pkl"
        with open(temp_path, 'rb') as f:
            data = CPU_Unpickler(f).load()
        scaler_data.append(data)
    all_data.append(torch.cat(scaler_data, 0).numpy())

all_data = np.array(all_data)
print(all_data.shape)

# Plot data
safe_path = current_path / "experiments/results/mimic/batchnorm/scaled/plots/"
if not os.path.exists(safe_path):
    os.makedirs(safe_path)

################## WORKING ########################################
# Separate scaled feature from the rest (other features were unaffected, so I leave them out)
age = all_data[:, :, 0]
# other = all_data[:, :, 1:]
# other = other.reshape((all_data.shape[0], -1))
print(age.shape)
# Gather statistics
age_mean = age.mean(axis=1)
age_std = age.std(axis=1)
age_max = age.max(axis=1)
age_min = age.min(axis=1)

# Try plotting a barplot of the scaled age after batchnorm
bins = np.arange(1, age.shape[0] + 1)
plt.bar(bins, age_mean, yerr=age_std)
plt.xticks(np.arange(1, len(scalers) + 1), scalers)
plt.legend()
plt.tight_layout()
plt.savefig(safe_path / 'out_histogram.png')
plt.clf()

# Try plotting a box
df = pd.DataFrame(data=age, columns=scalers)
sns.boxplot(x='scaler', y='batchnorm output', data=pd.melt(df))
# plt.boxplot(age, columns=scalers)
plt.tight_layout()
plt.savefig(safe_path / 'out_boxplot.png')
plt.clf()

# Try plotting a line plot with other metrics
print(age_min.shape, age_max.shape)
plt.plot(scalers, age_min, label='min')
plt.plot(scalers, age_max, label='max')
plt.plot(scalers, age_mean, label='mean')
plt.fill_between(scalers, age_mean - age_std, age_mean + age_std, alpha=0.2, label='mean')
plt.legend()
plt.tight_layout()
plt.savefig(safe_path / 'out_lineplot.png')
plt.clf()
# TODO: plot the metrics (in the correct directory not where I am loading from rn)
