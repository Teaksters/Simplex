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
# Separate scaled feature from the rest
age = all_data[:, :, 0]
# age = age.reshape((all_data.shape[0], -1))
other = all_data[:, :, 1:]
# other = other.reshape((all_data.shape[0], -1))

print(age.shape, other.shape)

# Try plotting a histogram of the features after batchnorm
age_mean = age.mean(axis=1)
age_std = age.std(axis=1)
other_mean = other.mean(axis=1)
other_std = other.std(axis=1)
print(age_mean.shape, other_mean.shape)
exit()

bins = np.arange(1, 191)
for i, data in enumerate(all_data_mean):
    plt.bar(bins, all_data_mean[i], yerr=all_data_std[i], label=scalers[i],
             alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig(safe_path / 'out_histogram.png')
plt.clf()

# Try plotting a box
columns = all_data.reshape((all_data.shape[0], -1))
plt.boxplot(columns)
plt.xticks(np.arange(1, len(scalers) + 1), scalers)
plt.tight_layout()
plt.savefig(safe_path / 'out_boxplot.png')

# TODO: plot the metrics (in the correct directory not where I am loading from rn)
