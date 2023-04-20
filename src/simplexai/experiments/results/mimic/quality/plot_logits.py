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

from simplexai.models.tabular_data import MortalityPredictor


class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-cv_list",
    nargs="+",
    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    help="The list of experiment cv identifiers to plot",
    type=int,
)
parser.add_argument(
    "-age_scalers",
    nargs='*',
    type=float,
    default=[1.0, 1.25, 1.5, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
    help="Scaling variable for sample ages"
)
args = parser.parse_args()
cv_list = args.cv_list
scalers = args.age_scalers
explainer_names = ["simplex"]
names_dict = {
    "simplex": "SimplEx"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rc("text", usetex=True)
params = {"text.latex.preamble": r"\usepackage{amsmath}"}
plt.rcParams.update(params)
representer_metrics = np.zeros((2, len(cv_list)))
current_path = Path.cwd()
load_path = current_path / "experiments" / "results" / "mimic" / "quality" / "scaled"

# Collect logit data
data = []
print(scalers, cv_list)
for scaler in scalers:
    data.append([])
    for cv in cv_list:
        corpus_data_path = load_path / str(scaler) / f"corpus_data_cv{cv}.pkl"
        with open(corpus_data_path, "rb") as f:
            corpus_data = CPU_Unpickler(f).load()
        logits = corpus_data[0]
        data[-1].append(logits)
    data[-1] = [logit.numpy() for l in data[-1] for logit in l]
data = np.array(data)

# Reduce logits to their vector length (norms)
logit_norms = np.empty(data.shape[:2])
for i_scale in range(len(data)):
    for i_cv in range(len(data[0])):
        logit_norms[i_scale, i_cv] = np.linalg.norm(data[i_scale, i_cv])

# plot logit norms into a histogram
if not os.path.exists('experiments/results/mimic/quality/logits/plots'):
    os.makedirs('experiments/results/mimic/quality/logits/plots')

plot_dict = {}
for i, scaler in enumerate(scalers):
    plot_dict[scaler] = list(logit_norms[i])
df = pd.DataFrame(plot_dict)
print(df)
df.boxplot(column=['1.0', '1.5', '2.0', '5.0', '10.0'])
safe_path = 'experiments/results/mimic/quality/logits/plots/logit_boxplot.png'
plt.savefig(safe_path)
