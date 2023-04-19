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

from simplexai.models.tabular_data import MortalityPredictor

parser = argparse.ArgumentParser()
parser.add_argument(
    "-cv_list",
    nargs="+",
    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    help="The list of experiment cv identifiers to plot",
    type=int,
)
parser.add_argument(
    "-k_list",
    nargs="+",
    default=[5, 10, 15],
    help="The list of active corpus members considered",
    type=int,
)
parser.add_argument(
    "-age_scalers",
    nargs='*',
    type=float,
    default=[1.0, 1.25, 1.5, 2.0, 5.0],
    help="Scaling variable for sample ages")

args = parser.parse_args()
cv_list = args.cv_list
n_keep_list = args.k_list
scalers = args.age_scalers
explainer_names = ["simplex"]
names_dict = {
    "simplex": "SimplEx"
}

line_styles = {"simplex": "-", "nn_uniform": "--", "nn_dist": ":"}
# MAYBE NEED COLORS HERE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rc("text", usetex=True)
params = {"text.latex.preamble": r"\usepackage{amsmath}"}
plt.rcParams.update(params)
representer_metrics = np.zeros((2, len(cv_list)))
current_path = Path.cwd()
load_path = current_path / "experiments" / "results" / "mimic" / "quality" / "scaled"

########### NEED TO TEST FROM HERE!!!!!!!!!!!!!!!!!!!!!!!!! ################

data = []
for scaler in scalers:
    scalers.append(scaler)
    data.append([])
    for cv in cv_list:
        corpus_data_path = load_path / str(scaler) / f"corpus_data_cv{cv}.pkl"
        with open(corpus_data_path, "rb") as f:
            corpus_data = pickle.load(f)
        logits = corpus_data[0]
        data[-1].append(logits)
        print(logits)
print(data)


exit()
for i, logit in enumerate(data):
    plt.hist(logit, label=scalers[i], alpha=0.3)
plt.savefig(...)


################## WORKING ########################################
# TODO: make histogram from differently scaled logits


safe_path = load_path / 'logits' / 'plots'
if not os.path.exists(safe_path):
    os.makedirs(safe_path)
