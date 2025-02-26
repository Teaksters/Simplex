import argparse
import os
import pickle as pkl
from pathlib import Path

import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import copy

from simplexai.explainers.nearest_neighbours import NearNeighLatent
from simplexai.explainers.representer import Representer
from simplexai.explainers.simplex import Simplex
from simplexai.models.tabular_data import MortalityPredictor
from simplexai.utils.schedulers import ExponentialScheduler
from simplexai.experiments.tabular_mimic import MimicDataset, \
                                                load_from_preprocessed, \
                                                load_age, \
                                                load_tabular_mimic



ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = Path('../Data/preprocessed')


def approximation_quality(
    cv: int = 0,
    age_scaler: float=1.,
    random_seed: int = 55,
    save_path: str = "experiments/results/mimic/batchnorm/scaled", # update here
    train_model: bool = True,
    train_data_only=False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.random.manual_seed(random_seed + cv)

    print(
        100 * "-"
        + "\n"
        + "Welcome in the approximation quality experiment for in hospital mortality. \n"
        f"Settings: random_seed = {random_seed} ; cv = {cv} ; device = {device}.\n"
        + 100 * "-"
    )

    # Define parameters
    n_epoch_model = 5
    log_interval = 100
    weight_decay = 1e-5
    corpus_size = 100
    test_size = 100
    n_keep_list = [10]  # NEED TO PICK A SUITABLE K
    reg_factor_init = 0.01
    reg_factor_final = 1.0
    n_epoch_simplex = 10000

    current_path = Path.cwd()
    save_path = current_path / save_path / str(age_scaler)  # Update here
    if not save_path.exists():
        print(f"Creating the saving directory {save_path}")
        os.makedirs(save_path)

    # Load the data
    X, y = load_tabular_mimic(random_seed=random_seed + cv)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_seed + cv, stratify=y
    )

    train_data = MimicDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=50, shuffle=True)
    test_data = MimicDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=50, shuffle=True)

    if train_model:
        # Create the model
        classifier = MortalityPredictor()
        classifier.to(device)
        optimizer = optim.Adam(classifier.parameters(), weight_decay=weight_decay)

        # Train the model
        print(100 * "-" + "\n" + "Now fitting the model. \n" + 100 * "-")
        train_losses = []
        train_counter = []
        test_losses = []

        def train(epoch):
            classifier.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.type(torch.LongTensor)
                target = target.to(device)
                optimizer.zero_grad()
                output = classifier(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    print(
                        f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                        f" ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                    )
                    train_losses.append(loss.item())
                    train_counter.append(
                        (batch_idx * 128) + ((epoch - 1) * len(train_loader.dataset))
                    )
                    torch.save(classifier.state_dict(), save_path / f"model_cv{cv}.pth")
                    torch.save(
                        optimizer.state_dict(), save_path / f"optimizer_cv{cv}.pth"
                    )

        def test():
            classifier.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device)
                    target = target.type(torch.LongTensor)
                    target = target.to(device)
                    output = classifier(data)
                    test_loss += F.nll_loss(output, target, reduction="sum").item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
            print(
                f"\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}"
                f"({100. * correct / len(test_loader.dataset):.0f}%)\n"
            )

        test()
        for epoch in range(1, n_epoch_model + 1):
            train(epoch)
            test()
        torch.save(classifier.state_dict(), save_path / f"model_cv{cv}.pth")
        torch.save(optimizer.state_dict(), save_path / f"optimizer_cv{cv}.pth")

    # Load model:
    classifier = MortalityPredictor()
    classifier.load_state_dict(torch.load(save_path / f"model_cv{cv}.pth"))
    classifier.to(device)
    classifier.eval()

    ########################## WORKING HERE TO ADJUST AGE ######################
    # Load data for the explainers
    print(100 * "-" + "\n" + "Now fitting the explainers. \n" + 100 * "-")

    explainer_names = ["simplex", "nn_uniform", "nn_dist"]

    corpus_loader = DataLoader(train_data, batch_size=corpus_size, shuffle=True)

    if train_data_only:
        test_loader = DataLoader(train_data, batch_size=test_size, shuffle=True)
    else:
        test_loader = DataLoader(test_data, batch_size=test_size, shuffle=True)
    corpus_examples = enumerate(corpus_loader)
    test_examples = enumerate(test_loader)
    batch_id_test, (test_data, test_targets) = next(test_examples)
    batch_id_corpus, (corpus_data, corpus_target) = next(corpus_examples)
    corpus_data = corpus_data.to(device).detach()

    # Experiment with age scaling
    corpus_data[:, 0] = corpus_data[:, 0] * age_scaler

    test_data = test_data.to(device).detach()
    corpus_batchnorm_out = classifier.cont_batchnorm_output(corpus_data).detach()

    ############################################################################

    # Save data:
    data_safe_path = save_path / f"BNorm_out_data_cv{cv}.pkl"
    with open(data_safe_path, "wb") as f:
        print(f"Saving corpus data in {data_safe_path}.")
        pkl.dump(corpus_batchnorm_out, f)

def main(experiment: str = "approximation_quality",
         cv: int = 0,
         age_scaler: float = 1.,
         corpus_scaler: float = 1.) -> None:
    if experiment == "approximation_quality":
        approximation_quality(cv=cv, age_scaler=age_scaler)
    else:
        raise ValueError(
            "The name of the experiment is not valid. "
            "Valid names are: approximation_quality, outlier_detection, outlier_detection2, corpus_size."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-experiment",
        type=str,
        default="approximation_quality",
        help="Experiment to perform",
    )
    parser.add_argument("-cv", type=int, default=0, help="Cross validation parameter")
    parser.add_argument("-age_scalers", nargs='*', type=float, default=[1.], help="Scaling variable for sample ages")
    parser.add_argument("-corpus_scalers", nargs='*', type=float, default=[1.], help="Scaling variable for corpus ages")
    args = parser.parse_args()
    for age_scaler in args.age_scalers:
        for corpus_scaler in args.corpus_scalers:
            main(args.experiment, args.cv, age_scaler, corpus_scaler)
