import argparse
import pickle as pkl
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score

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
    "-OOD_scalers",
    nargs="*",
    default=[2.0],
    help="The scalars used to rescale outliers to be OOD.",
    type=float,
)
parser.add_argument(
    "-corpus_scalers",
    nargs="*",
    default=['pregen', 1.0, 5.0],
    help="The scalars used to rescale outliers to be OOD.",
    type=float,
)

args = parser.parse_args()
cv_list = args.cv_list
OOD_scalers = args.OOD_scalers
corpus_scalers = args.corpus_scalers
current_path = Path.cwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rc("text", usetex=True)
params = {"text.latex.preamble": r"\usepackage{amsmath}"}
plt.rcParams.update(params)
test_size = 200

metrics = np.zeros((4, test_size, len(cv_list)))
accuracies = np.zeros((4, test_size, len(cv_list)))
n_inspected = [n for n in range(test_size)]

load_path = current_path / "experiments/results/mimic/outlier/scaled2"
safe_path = load_path / 'plots'
if not os.path.exists(safe_path):
    os.makedirs(safe_path)

means = []
stds = []
for scaler in OOD_scalers:
    for c_scaler in corpus_scalers:
        if c_scaler == 'pregen':
            current_path = load_path.parent / 'scaled3' / 'pregen'
        else:
            current_path = load_path / str(scaler) / str(c_scaler)
        for cv in cv_list:
            classifier = MortalityPredictor()
            classifier.load_state_dict(torch.load(current_path / f"model_cv{cv}.pth"))
            classifier.to(device)
            classifier.eval()
            with open(current_path / f"test_data_cv{cv}.pkl", "rb") as f:
                test_latent_reps, test_targets = pkl.load(f)
            with open(current_path / f"simplex_cv{cv}.pkl", "rb") as f:
                simplex = pkl.load(f)
            # with open(current_path / f"nn_dist_cv{cv}.pkl", "rb") as f:
            #     nn_dist = pkl.load(f)
            # with open(current_path / f"nn_uniform_cv{cv}.pkl", "rb") as f:
            #     nn_uniform = pkl.load(f)

            latents_true = test_latent_reps.to(device)
            test_predictions = torch.argmax(
                classifier.latent_to_presoftmax(latents_true), dim=-1
            )
            simplex_latent_approx = simplex.latent_approx().to(device)
            # nn_dist_latent_approx = nn_dist.latent_approx().to(device)
            # nn_uniform_latent_approx = nn_uniform.latent_approx().to(device)
            simplex_residuals = ((latents_true - simplex_latent_approx) ** 2).mean(dim=-1)
            # nn_dist_residuals = ((latents_true - nn_dist_latent_approx) ** 2).mean(dim=-1)
            # nn_uniform_residuals = ((latents_true - nn_uniform_latent_approx) ** 2).mean(dim=-1)
            counts_simplex = []
            # counts_nn_dist = []
            # counts_nn_uniform = []
            counts_random = []
            random_perm = torch.randperm(test_size)
            for k in range(simplex_residuals.shape[0]):
                _, simplex_top_id = torch.topk(simplex_residuals, k)
                # _, nn_dist_top_id = torch.topk(nn_dist_residuals, k)
                # _, nn_uniform_top_id = torch.topk(nn_uniform_residuals, k)
                random_id = random_perm[:k]
                count_simplex = torch.count_nonzero(simplex_top_id > 99).item()
                # count_nn_dist = torch.count_nonzero(nn_dist_top_id > 99).item()
                # count_nn_uniform = torch.count_nonzero(nn_uniform_top_id > 99).item()
                count_random = torch.count_nonzero(random_id > 99).item()
                simplex_selected_targets = np.delete(
                    test_targets.cpu().numpy(), simplex_top_id.cpu().numpy()
                )
                simplex_selected_predictions = np.delete(
                    test_predictions.cpu().numpy(), simplex_top_id.cpu().numpy()
                )
                # nn_dist_selected_targets = np.delete(
                #     test_targets.cpu().numpy(), nn_dist_top_id.cpu().numpy()
                # )
                # nn_dist_selected_predictions = np.delete(
                #     test_predictions.cpu().numpy(), nn_dist_top_id.cpu().numpy()
                # )
                # nn_uniform_selected_targets = np.delete(
                #     test_targets.cpu().numpy(), nn_uniform_top_id.cpu().numpy()
                # )
                # nn_uniform_selected_predictions = np.delete(
                #     test_predictions.cpu().numpy(), nn_uniform_top_id.cpu().numpy()
                # )
                random_selected_targets = np.delete(
                    test_targets.cpu().numpy(), random_id.cpu().numpy()
                )
                random_selected_predictions = np.delete(
                    test_predictions.cpu().numpy(), random_id.cpu().numpy()
                )
                accuracies[0, k, cv] = accuracy_score(
                    simplex_selected_targets, simplex_selected_predictions
                )
                # accuracies[1, k, cv] = accuracy_score(
                #     nn_dist_selected_targets, nn_dist_selected_predictions
                # )
                # accuracies[2, k, cv] = accuracy_score(
                #     nn_uniform_selected_targets, nn_uniform_selected_predictions
                # )
                accuracies[3, k, cv] = accuracy_score(
                    random_selected_targets, random_selected_predictions
                )
                counts_simplex.append(count_simplex)
                # counts_nn_dist.append(count_nn_dist)
                # counts_nn_uniform.append(count_nn_uniform)
                counts_random.append(count_random)
            metrics[0, :, cv] = counts_simplex
            # metrics[1, :, cv] = counts_nn_dist
            # metrics[2, :, cv] = counts_nn_uniform
            metrics[3, :, cv] = counts_random

        counts_ideal = [
            n if n < int(test_size / 2) else int(test_size / 2) for n in range(test_size)
        ]
        means.append(metrics[0].mean(axis=-1))
        stds.append(metrics[0].std(axis=-1))

colors = ['#b2182b','#ef8a62','#fddbc7','#d1e5f0','#67a9cf','#2166ac']

sns.set(font_scale=1.5)
sns.set_style("white")
sns.set_palette("colorblind")

plt.plot(n_inspected, counts_ideal, label="Maximal", color=colors[-1])

plt.plot(n_inspected, means[0], "-", label="Prototypical Corpus", color=colors[1])
plt.fill_between(n_inspected, means[0] - stds[0], means[0] + stds[0], alpha=0.3, color=colors[1])

plt.plot(n_inspected, means[1], "-", label="Familiar Corpus" + str(corpus_scalers[1]), color=colors[5])
plt.fill_between(n_inspected, means[1] - stds[1], means[1] + stds[1], alpha=0.3, color=colors[5])

plt.plot(n_inspected, means[2], "-", label="Unfamiliar Corpus (scaler: 5)" + str(corpus_scalers[2]), color=colors[2])
plt.fill_between(n_inspected, means[2] - stds[2], means[2] + stds[2], alpha=0.3, color=colors[2])

# plt.plot(n_inspected, means[2], "-", label="corpus Age x" + str(corpus_scalers[2]), color=colors[3])
# plt.fill_between(n_inspected, means[2] - stds[2], means[2] + stds[2], alpha=0.3, color=colors[3])

# plt.plot(n_inspected, means[3], "-", label="corpus Age x" + str(corpus_scalers[3]), color=colors[4])
# plt.fill_between(n_inspected, means[3] - stds[3], means[3] + stds[3], alpha=0.3, color=colors[4])

plt.plot(n_inspected, metrics[3].mean(axis=-1), "-.", label="Random", color=colors[0])
plt.fill_between(
    n_inspected,
    metrics[3].mean(axis=-1) - metrics[3].std(axis=-1),
    metrics[3].mean(axis=-1) + metrics[3].std(axis=-1),
    alpha=0.3,
    color=colors[0]
)
plt.xlabel("Number of samples inspected")
plt.ylabel("Number of outliers detected")
plt.legend(loc='upper left', fontsize='x-small')
plt.title("Outlier Detection Performance With Pregenerated Corpus")

plt.savefig(safe_path / "outlier_pregen.jpg", bbox_inches="tight")

print('succesfully saved plot at:\n', safe_path)
