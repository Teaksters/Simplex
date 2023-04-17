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

metric_names = ["r2_latent", "r2_output", "residual_latent", "residual_output"]
results_df = pd.DataFrame(
    columns=[
        "explainer",
        "n_keep",
        "cv",
        "scaler",
        "logit",
    ]
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rc("text", usetex=True)
params = {"text.latex.preamble": r"\usepackage{amsmath}"}
plt.rcParams.update(params)
representer_metrics = np.zeros((2, len(cv_list)))
current_path = Path.cwd()
load_path = current_path / "experiments/results/mimic/quality/scaled/"

for scaler in scalers:
    for cv in cv_list:
        classifier = MortalityPredictor(n_cont=1, input_feature_num=26)
        classifier.load_state_dict(torch.load(load_path / str(scaler) / f"model_cv{cv}.pth"))
        classifier.to(device)
        classifier.eval()
        for n_keep in n_keep_list:
            for explainer_name in explainer_names:
                with open(load_path / str(scaler) / f"{explainer_name}_cv{cv}_n{n_keep}.pkl", "rb") as f:
                    explainer = pkl.load(f)
                explainer.to(device)
                corpus_logits = explainer.corpus_latent_reps
                corpus_logits = corpus_logits.norm(dim=1, p=0) # TODO: check if dimensionality is correct
                results_df = pd.concat(
                    [
                        results_df,
                        pd.DataFrame.from_dict(
                            {
                                "explainer": [explainer_name],
                                "n_keep": [n_keep],
                                "cv": [cv],
                                "scaler": [scaler],
                                "logit": [corpus_logits],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

# print(results_df.loc[(results_df['explainer'] == 'simplex') & \
#                      (results_df['n_keep'] == 5) & \
#                      (results_df['scaler'] == 1.0)])

################## WORKING ########################################
# TODO: make histogram from differently scaled logits


safe_path = load_path / 'logits' / 'plots'
if not os.path.exists(safe_path):
    os.makedirs(safe_path)
