import argparse
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import torch

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
explainer_names = ["simplex", "nn_uniform", "nn_dist"]
names_dict = {
    "simplex": "SimplEx",
    "nn_uniform": "KNN Uniform",
    "nn_dist": "KNN Distance",
}

line_styles = {"simplex": "-", "nn_uniform": "--", "nn_dist": ":"}
# MAYBE NEED COLORS HERE

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
load_path = current_path / "experiments/results/mimic/quality/scaled"

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
                                "scaler": [scaler]
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
# mean_df = results_df.groupby(["explainer", "n_keep", "scaler"]).aggregate("mean").unstack(level=0)
# std_df = results_df.groupby(["explainer", "n_keep", "scaler"]).aggregate("std").unstack(level=0)
# min_df = results_df.groupby(["explainer", "n_keep", "scaler"]).aggregate("min").unstack(level=0)
# max_df = results_df.groupby(["explainer", "n_keep", "scaler"]).aggregate("max").unstack(level=0)
# q1_df = results_df.groupby(["explainer", "n_keep", "scaler"]).quantile(0.25).unstack(level=0)
# q3_df = results_df.groupby(["explainer", "n_keep", "scaler"]).quantile(0.75).unstack(level=0)

# print(results_df.loc[(results_df['explainer'] == 'simplex') & \
#                      (results_df['n_keep'] == 5) & \
#                      (results_df['scaler'] == 1.0)])

################## WORKING ########################################
# TODO: make a boxplot with 3 groups (K) of ascending scalers (5)
for m, metric_name in enumerate(metric_names):
    for explainer_name in explainer_names:
        plt.figure(m + 1) # I want a seperate plot for each explainer AND metric
        data = []
        input = []
        Ks = []
        for k, K in enumerate(n_keep_list):
            data.append([])
            input.append([])
            Ks.append(K)
            for scaler in scalers:
                temp_data = results_df.loc[(results_df['explainer'] == explainer_name) & \
                                           (results_df['n_keep'] == K) & \
                                           (results_df['scaler'] == scaler)]
                data[k].append(temp_data[metric_name]) # Maybe this is wrong..?
                input[k].append(scaler)

            plt.boxplot(data[k],
                        positions=np.array(xrange(len(data[k])))*5.0-0.4,
                        widths=0.6,
                        label=n_keep_list[k])
        plt.ylim(0, 1)
        plt.xticks(xrange(0, len(n_keep_list) * 5, 5), n_keep_list)
        plt.legend()
        plt.tight_layout()
        plt.savefig('test.jpg')
        print(data)
        print(input)
        exit()

# data_a = [[1,2,5], [5,7,2,2,5], [7,2,5]]
# data_b = [[6,4,2], [1,2,5,3,2], [2,3,5,1]]

# ticks = ['A', 'B', 'C']

# def set_box_color(bp, color):
#     plt.setp(bp['boxes'], color=color)
#     plt.setp(bp['whiskers'], color=color)
#     plt.setp(bp['caps'], color=color)
#     plt.setp(bp['medians'], color=color)

# plt.figure()
#
# bpl = plt.boxplot(data_a, positions=np.array(xrange(len(data_a)))*2.0-0.4, sym='', widths=0.6)
# bpr = plt.boxplot(data_b, positions=np.array(xrange(len(data_b)))*2.0+0.4, sym='', widths=0.6)
# set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
# set_box_color(bpr, '#2C7BB6')

# draw temporary red and blue lines and use them to create a legend
# plt.plot([], c='#D7191C', label='Apples')
# plt.plot([], c='#2C7BB6', label='Oranges')
# plt.legend()
#
# plt.xticks(xrange(0, len(ticks) * 2, 2), ticks)
# plt.xlim(-2, len(ticks)*2)
# plt.ylim(0, 8)
# plt.tight_layout()
# plt.savefig('boxcompare.png')

# plt.figure(1)
# plt.xlabel(r"$K$")
# plt.ylabel(r"$R^2_{\mathcal{H}}$")
# plt.ylim(top=1.0)
# plt.legend()
# plt.savefig(load_path / "r2_latent.pdf", bbox_inches="tight")
# plt.figure(2)
# plt.xlabel(r"$K$")
# plt.ylabel(r"$R^2_{\mathcal{Y}}$")
# plt.ylim(top=1.0)
# plt.legend()
# plt.savefig(load_path / "r2_output.pdf", bbox_inches="tight")
# plt.figure(3)
# plt.xlabel(r"$K$")
# plt.ylabel(r"$\| \hat{\boldsymbol{h}} - \boldsymbol{h} \| $")
# plt.legend()
# plt.savefig(load_path / "residual_latent.pdf", bbox_inches="tight")
# plt.figure(4)
# plt.xlabel(r"$K$")
# plt.ylabel(r"$\| \hat{\boldsymbol{y}} - \boldsymbol{y} \| $")
# plt.legend()
# plt.savefig(load_path / "residual_output.pdf", bbox_inches="tight")
#
# print(
#     f"Representer metrics: r2_output = {representer_metrics[0].mean():.2g} +/- {representer_metrics[0].std():.2g}"
#     f" ; residual_output = {representer_metrics[1].mean():.2g} +/- {representer_metrics[1].std():.2g}"
# )