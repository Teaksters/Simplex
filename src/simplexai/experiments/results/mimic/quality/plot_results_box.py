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
    default=[10],
    help="The list of active corpus members considered",
    type=int,
)
parser.add_argument(
    "-age_scalers",
    nargs='*',
    type=float,
    default=[1.0, 1.25, 1.5, 2.0, 5.0],
    help="Scaling variable for sample ages"
)

args = parser.parse_args()
cv_list = args.cv_list
n_keep_list = args.k_list
scalers = args.age_scalers
explainer_names = ["simplex", "nn_uniform", "nn_dist"]
names_dict = {
    "simplex": "SimplEx",
    "nn_uniform": "KNN Uniform",
    "nn_dist": "KNN Distance",
}
line_styles = {"simplex": "-", "nn_uniform": "--", "nn_dist": ":"}
metric_names = ["r2_latent", "r2_output", "residual_latent", "residual_output"]
results_df = pd.DataFrame(
    columns=[
        "explainer",
        "n_keep",
        "cv",
        "r2_latent",
        "r2_output",
        "residual_latent",
        "residual_output",
        "scaler",
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
                latent_rep_approx = explainer.latent_approx()
                latent_rep_true = explainer.test_latent_reps
                output_approx = classifier.latent_to_presoftmax(latent_rep_approx).detach()
                output_true = classifier.latent_to_presoftmax(latent_rep_true).detach()
                latent_r2_score = sklearn.metrics.r2_score(
                    latent_rep_true.cpu().numpy(), latent_rep_approx.cpu().numpy()
                )
                output_r2_score = sklearn.metrics.r2_score(
                    output_true.cpu().numpy(), output_approx.cpu().numpy()
                )
                residual_latent = torch.sqrt(
                    ((latent_rep_true - latent_rep_approx) ** 2).mean()
                ).item()
                residual_output = torch.sqrt(
                    ((output_true - output_approx) ** 2).mean()
                ).item()
                results_df = pd.concat(
                    [
                        results_df,
                        pd.DataFrame.from_dict(
                            {
                                "explainer": [explainer_name],
                                "n_keep": [n_keep],
                                "cv": [cv],
                                "r2_latent": [latent_r2_score],
                                "r2_output": [output_r2_score],
                                "residual_latent": [residual_latent],
                                "residual_output": [residual_output],
                                "scaler": [scaler],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
        with open(load_path / str(scaler) / f"representer_cv{cv}.pkl", "rb") as f:
            representer = pkl.load(f)
        representer.to(device)
        latent_rep_true = representer.test_latent_reps
        output_approx = representer.output_approx()
        output_true = classifier.latent_to_presoftmax(latent_rep_true).detach()
        representer_metrics[0, cv] = sklearn.metrics.r2_score(
            output_true.cpu().numpy(), output_approx.cpu().numpy()
        )
        representer_metrics[1, cv] = torch.sqrt(
            ((output_true - output_approx) ** 2).mean() / (output_true**2).mean()
        ).item()

sns.set(font_scale=1.5)
sns.set_style("white")
sns.set_palette("colorblind")

for m, metric_name in enumerate(metric_names):
    plt.figure(m + 1)
    data = []
    for scaler in scalers: # Only need simplex
        temp_data = results_df.loc[(results_df['explainer'] == "simplex") & \
                                   (results_df['n_keep'] == 10) & \
                                   (results_df['scaler'] == scaler)]
        data.append(list(temp_data[metric_name])) # Maybe this is wrong..?
    plt.boxplot(data)
    plt.xticks([i for i in range(1, len(scalers) + 1)], scalers)
    plt.set_yscale('log')
    plt.set_xscale('log')

safe_path = load_path / 'plots3/'
if not os.path.exists(safe_path):
    os.makedirs(safe_path)

plt.figure(1)
plt.xlabel(r"$age scaling factor$")
plt.ylabel(r"$R^2_{\mathcal{H}}$")
plt.ylim(top=1.0)
plt.legend()
plt.savefig(safe_path / "r2_latent.pdf", bbox_inches="tight")
plt.figure(2)
plt.xlabel(r"$age scaling factor$")
plt.ylabel(r"$R^2_{\mathcal{Y}}$")
plt.ylim(top=1.0)
plt.legend()
plt.savefig(safe_path / "r2_output.pdf", bbox_inches="tight")
plt.figure(3)
plt.xlabel(r"$age scaling factor$")
plt.ylabel(r"$\| \hat{\boldsymbol{h}} - \boldsymbol{h} \| $")
plt.legend()
plt.savefig(safe_path / "residual_latent.pdf", bbox_inches="tight")
plt.figure(4)
plt.xlabel(r"$age scaling factor$")
plt.ylabel(r"$\| \hat{\boldsymbol{y}} - \boldsymbol{y} \| $")
plt.legend()
plt.savefig(safe_path / "residual_output.pdf", bbox_inches="tight")

print(
    f"Representer metrics: r2_output = {representer_metrics[0].mean():.2g} +/- {representer_metrics[0].std():.2g}"
    f" ; residual_output = {representer_metrics[1].mean():.2g} +/- {representer_metrics[1].std():.2g}"
)
