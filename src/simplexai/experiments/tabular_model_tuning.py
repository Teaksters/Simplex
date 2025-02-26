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
DATA_DIR = '../Data/preprocessed'


def approximation_quality(
    args,
    cv: int = 0,
    random_seed: int = 55,
    save_path: str = "experiments/results/mimic/quality/tuning", # update here
    train_model: bool = True,
    train_data_only=False,
    epochs=5
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
    n_epoch_model = epochs
    log_interval = args.log_interval
    weight_decay = args.weight_decay
    corpus_size = 100
    test_size = 100
    n_keep_list = [10]  # NEED TO PICK A SUITABLE K
    reg_factor_init = 0.01
    reg_factor_final = 1.0
    n_epoch_simplex = 10000

    current_path = Path.cwd()
    save_path = current_path / save_path / 'epochs' / str(epochs)  # Update here
    if not save_path.exists():
        print(f"Creating the saving directory {save_path}")
        os.makedirs(save_path)

    # Load the data and split into train, val and test sets
    X, y = load_tabular_mimic(random_seed=random_seed + cv)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_seed + cv, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=random_seed + cv, stratify=y_test
    )

    train_data = MimicDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=50, shuffle=True)
    val_data = MimicDataset(X_val, y_val)
    val_loader = DataLoader(val_data, batch_size=50, shuffle=True)
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
        test_accs = []
        test_aucs = []

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

        def test(data_loader):
            classifier.eval()
            test_loss = 0
            correct = 0
            probas = []
            labels = []
            with torch.no_grad():
                for data, target in data_loader:
                    data = data.to(device)
                    target = target.type(torch.LongTensor)
                    target = target.to(device)
                    output = classifier(data)
                    probs = classifier.probabilities(data)
                    test_loss += F.nll_loss(output, target, reduction="sum").item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
                    probas.append(probs)
                    labels.append(target)

            probas = torch.cat(probas, 0)
            probas = probas.cpu().detach().numpy()
            labels = torch.cat(labels, 0)
            labels = labels.cpu().detach().numpy().flatten()

            probas = probas[:, 1] # Needs to provide the proba with the class with the greater label
            auc_score = sklearn.metrics.roc_auc_score(labels, probas)

            test_loss /= len(data_loader.dataset)
            test_losses.append(test_loss)
            test_accs = [correct / len(data_loader.dataset)]
            print(
                f"\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)}, AUC: {auc_score:.4f}"
                f"({100. * correct / len(data_loader.dataset):.0f}%)\n"
            )
            return test_accs, auc_score

        test_acc, auc_score = test(val_loader)
        test_accs.append([test_acc[0].item()])
        test_aucs.append(auc_score)
        for epoch in range(1, n_epoch_model + 1):
            train(epoch) # writes to performance data (bad coding practice I know..)
            test_acc, auc_score = test(val_loader)
            test_accs[-1].append(test_acc[0].item())
            test_aucs.append(auc_score)
        torch.save(classifier.state_dict(), save_path / f"model_cv{cv}.pth")
        torch.save(optimizer.state_dict(), save_path / f"optimizer_cv{cv}.pth")

        # Store losses for tuning purposes
        performance_data = [train_losses, train_counter , test_losses, test_accs, test_aucs]
        file = open(save_path / f"val_performance_cv{cv}.pkl", 'wb')
        pkl.dump(performance_data, file)

        # Test model on unseen test set
        test_acc, auc_score = test(test_loader)
        file = open(save_path / f"test_performance_cv{cv}.pkl", 'wb')
        pkl.dump([test_acc, auc_score], file)

def main(args) -> None:
    for epoch in args.epochs:
        approximation_quality(args, cv=args.cv, epochs=epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-experiment",
        type=str,
        default="approximation_quality",
        help="Experiment to perform",
    )
    parser.add_argument("-cv", type=int, default=0, help="Cross validation parameter")
    parser.add_argument("-epochs", nargs='*', type=int, default=[5], help="Scaling variable for corpus ages")
    parser.add_argument("-log_interval", type=int, default=10, help="Train performance logging intervals.")
    parser.add_argument("-weight_decay", type=float, default=1e-5, help="Weight decay used for training the model.")
    # parser.add_argument("-simplex_epochs", type=int, default=10000, help="Epochs to train simplex.")
    # parser.add_argument("-Ks", nargs='*', type=int, default=[10], help="Corpus size.")
    args = parser.parse_args()

    main(args)
