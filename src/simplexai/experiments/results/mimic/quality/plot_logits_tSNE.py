import argparse
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
from sklearn.manifold import TSNE
import torch
import os
import io

from simplexai.models.tabular_data import MortalityPredictor


class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def clean_alt_list(list_):
    list_ = list_.replace(', ', '","')
    list_ = list_.replace('[', '["')
    list_ = list_.replace(']', '"]')
    return list_

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
    data[-1] = [list(logit.numpy()) for l in data[-1] for logit in l]
# data = np.array(data)

logits = []
ys = []
for i, scaler in enumerate(scalers):
    logits += list(data[i])
    ys += [scaler] * len(data[i])

# Reduce logits to 2 dimensional space using tSNE reduction
tsne = TSNE(n_components=2, verbose=1, random_state=42, perplexity=20.0)
tsne_z = tsne.fit_transform(np.array(logits))

df = pd.DataFrame()
df['y'] = ys
df['x1'] = tsne_z[:, 0]
df['x2'] = tsne_z[:, 1]

# Only take the wanted scalers
plotted_scalers = [1.0, 100.0]
df = df[df['y'].isin(plotted_scalers)]

# plot tSNE projection as scatterplot
if not os.path.exists('experiments/results/mimic/quality/logits/plots'):
    os.makedirs('experiments/results/mimic/quality/logits/plots')

# https://www.datatechnotes.com/2020/11/tsne-visualization-example-in-python.html
sns.scatterplot(x='x1', y='x2', hue=df.y.tolist(),
                palette=sns.color_palette("hls", len(plotted_scalers)),
                data=df).set(title='tSNE projection logits')
safe_path = 'experiments/results/mimic/quality/logits/plots/logit_tSNE.png'
plt.savefig(safe_path)
