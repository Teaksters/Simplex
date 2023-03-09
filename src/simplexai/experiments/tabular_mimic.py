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

from simplexai.explainers.nearest_neighbours import NearNeighLatent
from simplexai.explainers.representer import Representer
from simplexai.explainers.simplex import Simplex
from simplexai.models.tabular_data import MortalityPredictor
from simplexai.utils.schedulers import ExponentialScheduler

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = '../Data/preprocessed'


class MimicDataset(Dataset):
    def __init__(self, X, y=None) -> None:
        self.X = X
        self.y = y.astype(int)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple:
        data = torch.tensor(self.X.iloc[i, :], dtype=torch.float32)
        target = self.y.iloc[i]
        return data, target

def load_from_preprocessed(dir):
    # Reads and concatenates the train and test data into one dataframe
    data = [os.path.join(dir, sub_dir, 'listfile.csv')
                  for sub_dir in os.listdir(dir)]
    dfs = [pd.read_csv(x) for x in data]
    df = pd.concat(dfs)
    return df

def load_age(): # COULD BE USED FOR MORE VALUES LATER BY NOT DROPPING THOSE COLS
    drop_cols = ['HADM_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'DBSOURCE', 'INTIME',
                 'OUTTIME', 'LOS', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME',
                 'ETHNICITY', 'DIAGNOSIS', 'GENDER', 'DOB', 'DOD',
                 'MORTALITY_INUNIT', 'MORTALITY', 'MORTALITY_INHOSPITAL']
    full_df = pd.read_csv(os.path.join(DATA_DIR, 'all_stays.csv'))
    full_df.sort_values(['SUBJECT_ID', 'ICUSTAY_ID'])
    # Adhere to data format of other dataframes
    general_df = full_df[full_df.duplicated('SUBJECT_ID') == False]
    general_df['SUBJECT_ID'] = general_df['SUBJECT_ID'].astype(str) + "_episode1_timeseries.csv"

    duplicates_df = full_df[full_df.duplicated('SUBJECT_ID') == True]
    i = 2
    while not duplicates_df.empty:
        # update general df
        episode_str = "_episode" + str(i) + "_timeseries.csv"
        temp_df = duplicates_df[duplicates_df.duplicated('SUBJECT_ID') == False]
        temp_df['SUBJECT_ID'] = temp_df['SUBJECT_ID'].astype(str) + episode_str
        general_df = pd.concat([general_df, temp_df])
        # prepare for next round
        duplicates_df = duplicates_df[duplicates_df.duplicated('SUBJECT_ID') == True]
        i += 1

    general_df.rename(columns={'SUBJECT_ID': 'stay'}, inplace=True)
    general_df.drop(columns=drop_cols, inplace=True)
    return general_df


def load_tabular_mimic(random_seed: int = 42) -> tuple:
    # Specify undesired columns
    drop_cols = ['stay', 'period_length']

    # Load MIMIC-III data into panda dataframes
    label_dir = os.path.join(DATA_DIR, 'in-hospital-mortality')
    label_df = load_from_preprocessed(label_dir)
    feature_dir = os.path.join(DATA_DIR, 'phenotyping')
    feature_df = load_from_preprocessed(feature_dir)
    age_df = load_age()

    # Merge data into workable complete format
    data_df = pd.merge(label_df, feature_df, on='stay')
    data_df = pd.merge(data_df, age_df, on='stay')
    data_df.drop(columns=drop_cols, inplace=True)

    ##################### OPTIONAL ######################################
    ### Balance data set for even amount of survivors and mortalities ###
    #####################################################################
    # mask = data_df[label] is True
    # df_dead = data_df[mask]
    # df_survive = data_df[~mask]
    # data_df = pd.concat(
    #     [
    #         df_dead.sample(2500, random_state=random_seed),
    #         df_survive.sample(2500, random_state=random_seed),
    #     ]
    # )
    ############################################################################

    df = sklearn.utils.shuffle(data_df, random_state=random_seed)
    df = df.reset_index(drop=True)
    features, labels = df.loc[:, df.columns != 'y_true'], df['y_true']
    return features, labels


def load_cutract(random_seed: int = 42) -> tuple:
    features = [
        "age",
        "psa",
        "comorbidities",
        "treatment_CM",
        "treatment_Primary hormone therapy",
        "treatment_Radical Therapy-RDx",
        "treatment_Radical therapy-Sx",
        "grade_1.0",
        "grade_2.0",
        "grade_3.0",
        "grade_4.0",
        "grade_5.0",
        "stage_1",
        "stage_2",
        "stage_3",
        "stage_4",
        "gleason1_1",
        "gleason1_2",
        "gleason1_3",
        "gleason1_4",
        "gleason1_5",
        "gleason2_1",
        "gleason2_2",
        "gleason2_3",
        "gleason2_4",
        "gleason2_5",
    ]
    label = "mortCancer"
    df = pd.read_csv(
        os.path.abspath(
            os.path.join(ROOT_DIR, "../Data/")
        )
    )
    mask = df[label] is True
    df_dead = df[mask]
    df_survive = df[~mask]
    df = pd.concat(
        [
            df_dead.sample(1000, random_state=random_seed),
            df_survive.sample(1000, random_state=random_seed),
        ]
    )
    df = sklearn.utils.shuffle(df, random_state=random_seed)
    df = df.reset_index(drop=True)
    return df[features], df[label]

def approximation_quality(
    cv: int = 0,
    random_seed: int = 55,
    save_path: str = "experiments/results/mimic/quality/",
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
    n_keep_list = [2, 5, 10, 50]
    reg_factor_init = 0.01
    reg_factor_final = 1.0
    n_epoch_simplex = 10000

    current_path = Path.cwd()
    save_path = current_path / save_path
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

    ########################## WORKING ######################
    if train_model:
        # Create the model
        classifier = MortalityPredictor(n_cont=1, input_feature_num=26)  # WORKING ON THIS
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

                    #######################################33
                    print(data)
                    exit()
                    #########################################
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
    classifier = MortalityPredictor(n_cont=1, input_feature_num=26)
    classifier.load_state_dict(torch.load(save_path / f"model_cv{cv}.pth"))
    classifier.to(device)
    classifier.eval()

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
    test_data = test_data.to(device).detach()
    corpus_latent_reps = classifier.latent_representation(corpus_data).detach()
    corpus_probas = classifier.probabilities(corpus_data).detach()
    corpus_true_classes = torch.zeros(corpus_probas.shape, device=device)
    corpus_true_classes[
        torch.arange(corpus_size), corpus_target.type(torch.LongTensor)
    ] = 1
    test_latent_reps = classifier.latent_representation(test_data).detach()

    # Save data:
    corpus_data_path = save_path / f"corpus_data_cv{cv}.pkl"
    with open(corpus_data_path, "wb") as f:
        print(f"Saving corpus data in {corpus_data_path}.")
        pkl.dump([corpus_latent_reps, corpus_probas, corpus_true_classes], f)
    test_data_path = save_path / f"test_data_cv{cv}.pkl"
    with open(test_data_path, "wb") as f:
        print(f"Saving test data in {test_data_path}.")
        pkl.dump([test_latent_reps, test_targets], f)

    # Fit the explainers
    for n_keep in n_keep_list:
        print(30 * "-" + f"n_keep = {n_keep}" + 30 * "-")
        explainers = []
        # Fit SimplEx:
        reg_factor_scheduler = ExponentialScheduler(
            reg_factor_init, reg_factor_final, n_epoch_simplex
        )
        simplex = Simplex(
            corpus_examples=corpus_data, corpus_latent_reps=corpus_latent_reps
        )
        simplex.fit(
            test_examples=test_data,
            test_latent_reps=test_latent_reps,
            n_epoch=n_epoch_simplex,
            reg_factor=reg_factor_init,
            n_keep=n_keep,
            reg_factor_scheduler=reg_factor_scheduler,
        )
        explainers.append(simplex)

        # Fit nearest neighbors:
        nn_uniform = NearNeighLatent(
            corpus_examples=corpus_data, corpus_latent_reps=corpus_latent_reps
        )
        nn_uniform.fit(
            test_examples=test_data, test_latent_reps=test_latent_reps, n_keep=n_keep
        )
        explainers.append(nn_uniform)
        nn_dist = NearNeighLatent(
            corpus_examples=corpus_data,
            corpus_latent_reps=corpus_latent_reps,
            weights_type="distance",
        )
        nn_dist.fit(
            test_examples=test_data, test_latent_reps=test_latent_reps, n_keep=n_keep
        )
        explainers.append(nn_dist)

        # Save explainers:
        for explainer, explainer_name in zip(explainers, explainer_names):
            explainer_path = save_path / f"{explainer_name}_cv{cv}_n{n_keep}.pkl"
            with open(explainer_path, "wb") as f:
                print(f"Saving {explainer_name} decomposition in {explainer_path}.")
                pkl.dump(explainer, f)
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
            print(
                f"{explainer_name} latent r2: {latent_r2_score:.2g} ; output r2 = {output_r2_score:.2g}."
            )

    # Fit the representer:
    representer = Representer(
        corpus_latent_reps=corpus_latent_reps,
        corpus_probas=corpus_probas,
        corpus_true_classes=corpus_true_classes,
        reg_factor=weight_decay,
    )
    representer.fit(test_latent_reps=test_latent_reps)
    explainer_path = save_path / f"representer_cv{cv}.pkl"
    with open(explainer_path, "wb") as f:
        print(f"Saving representer decomposition in {explainer_path}.")
        pkl.dump(representer, f)
    latent_rep_true = representer.test_latent_reps
    output_true = classifier.latent_to_presoftmax(latent_rep_true).detach()
    output_approx = representer.output_approx()
    output_r2_score = sklearn.metrics.r2_score(
        output_true.cpu().numpy(), output_approx.cpu().numpy()
    )
    print(f"representer output r2 = {output_r2_score:.2g}.")

########################## HAVEN'T TESTED FUNCTIONALITY ########################
def outlier_detection(
    cv: int = 0,
    random_seed: int = 42,
    save_path: str = "experiments/results/prostate/outlier/",
    train_model: bool = True,
) -> None:
    torch.random.manual_seed(random_seed + cv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        100 * "-"
        + "\n"
        + "Welcome in the outlier detection experiment for Prostate Cancer. \n"
        f"Settings: random_seed = {random_seed} ; cv = {cv} ; device = {device}.\n"
        + 100 * "-"
    )

    # Create saving directory if inexistent
    current_path = Path.cwd()
    save_path = current_path / save_path
    if not save_path.exists():
        print(f"Creating the saving directory {save_path}")
        os.makedirs(save_path)

    # Define parameters
    n_epoch_model = 5
    log_interval = 100
    weight_decay = 1e-5
    corpus_size = 100
    test_size = 100
    n_epoch_simplex = 10000

    # Load the data
    X, y = load_tabular_mimic(random_seed=random_seed + cv)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_seed + cv, stratify=y
    )
    train_data = MimicDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=50, shuffle=True)
    test_data = MimicDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=50, shuffle=True)
    X_cutract, y_cutract = load_cutract(random_seed=random_seed + cv)
    cutract_data = MimicDataset(X_cutract, y_cutract)
    cutract_loader = DataLoader(cutract_data, batch_size=test_size, shuffle=True)

    # Training a model, save it
    if train_model:
        # Create the model
        classifier = MortalityPredictor(n_cont=3)
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
    classifier = MortalityPredictor(n_cont=3)
    classifier.load_state_dict(torch.load(save_path / f"model_cv{cv}.pth"))
    classifier.to(device)
    classifier.eval()

    # Load data for the explainers
    print(100 * "-" + "\n" + "Now fitting the explainer. \n" + 100 * "-")

    corpus_loader = DataLoader(train_data, batch_size=corpus_size, shuffle=True)
    seer_loader = DataLoader(test_data, batch_size=test_size, shuffle=True)

    corpus_examples = enumerate(corpus_loader)
    seer_examples = enumerate(seer_loader)
    cutract_examples = enumerate(cutract_loader)
    _, (corpus_features, corpus_target) = next(corpus_examples)
    _, (seer_features, seer_targets) = next(seer_examples)
    _, (cutract_features, cutract_targets) = next(cutract_examples)
    corpus_features = corpus_features.to(device).detach()
    seer_features = seer_features.to(device).detach()
    cutract_features = cutract_features.to(device).detach()
    test_features = torch.cat([seer_features, cutract_features], dim=0)
    test_targets = torch.cat([seer_targets, cutract_targets], dim=0)
    corpus_latent_reps = classifier.latent_representation(corpus_features).detach()
    corpus_probas = classifier.probabilities(corpus_features).detach()
    corpus_true_classes = torch.zeros(corpus_probas.shape, device=device)
    corpus_true_classes[
        torch.arange(corpus_size), corpus_target.type(torch.LongTensor)
    ] = 1
    test_latent_reps = classifier.latent_representation(test_features).detach()

    # Save data:
    corpus_data_path = save_path / f"corpus_data_cv{cv}.pkl"
    with open(corpus_data_path, "wb") as f:
        print(f"Saving corpus data in {corpus_data_path}.")
        pkl.dump([corpus_latent_reps, corpus_probas, corpus_true_classes], f)
    test_data_path = save_path / f"test_data_cv{cv}.pkl"
    with open(test_data_path, "wb") as f:
        print(f"Saving test data in {test_data_path}.")
        pkl.dump([test_latent_reps, test_targets], f)

    # Fit explainers:
    simplex = Simplex(
        corpus_examples=corpus_features, corpus_latent_reps=corpus_latent_reps
    )
    simplex.fit(
        test_examples=test_features,
        test_latent_reps=test_latent_reps,
        n_epoch=n_epoch_simplex,
        reg_factor=0,
        n_keep=corpus_features.shape[0],
    )
    explainer_path = save_path / f"simplex_cv{cv}.pkl"
    with open(explainer_path, "wb") as f:
        print(f"Saving simplex decomposition in {explainer_path}.")
        pkl.dump(simplex, f)

    nn_uniform = NearNeighLatent(
        corpus_examples=corpus_features, corpus_latent_reps=corpus_latent_reps
    )
    nn_uniform.fit(test_features, test_latent_reps, n_keep=7)
    nn_dist = NearNeighLatent(
        corpus_examples=corpus_features,
        corpus_latent_reps=corpus_latent_reps,
        weights_type="distance",
    )
    nn_dist.fit(test_features, test_latent_reps, n_keep=7)
    explainer_path = save_path / f"nn_dist_cv{cv}.pkl"
    with open(explainer_path, "wb") as f:
        print(f"Saving nn_dist decomposition in {explainer_path}.")
        pkl.dump(nn_dist, f)
    explainer_path = save_path / f"nn_uniform_cv{cv}.pkl"
    with open(explainer_path, "wb") as f:
        print(f"Saving nn_uniform decomposition in {explainer_path}.")
        pkl.dump(nn_uniform, f)
 ###############################################################################

########################## HAVEN'T TESTED FUNCTIONALITY ########################
def corpus_size_effect(random_seed: int = 42) -> None:
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        100 * "-"
        + "\n"
        + "Welcome in the outlier detection experiment for Prostate Cancer. \n"
        f"Settings: random_seed = {random_seed} ; device = {device}.\n" + 100 * "-"
    )

    corpus_sizes = [50, 100, 500, 1000]
    test_size = 100
    residuals = torch.zeros(len(corpus_sizes), 4)
    current_directory = Path.cwd()

    for cv in range(4):
        if not (
            current_directory / "results/prostate/quality" / f"model_cv{cv}.pth"
        ).exists():
            raise RuntimeError(
                f"The approximation quality experiment for cv = {cv} should be run first."
            )
        print(25 * "=" + f"Now working with cv = {cv}." + 25 * "=")
        # Load model:
        classifier = MortalityPredictor(n_cont=3)
        classifier.load_state_dict(
            torch.load(
                current_directory / "results/prostate/quality" / f"model_cv{cv}.pth"
            )
        )
        classifier.to(device)
        classifier.eval()

        # Load the data
        X, y = load_tabular_mimic(random_seed=random_seed + cv)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=random_seed + cv, stratify=y
        )
        train_data = MimicDataset(X_train, y_train)
        test_data = MimicDataset(X_test, y_test)

        corpus_loader = DataLoader(
            train_data, batch_size=corpus_sizes[-1], shuffle=True
        )
        test_loader = DataLoader(test_data, batch_size=test_size, shuffle=True)

        Corpus_inputs, _ = next(iter(corpus_loader))
        test_inputs, _ = next(iter(test_loader))
        test_latents = classifier.latent_representation(test_inputs.to(device)).detach()

        for id_size, corpus_size in enumerate(corpus_sizes):
            print(f"Now fitting a corpus of size {corpus_size}.")
            corpus_inputs = Corpus_inputs[torch.randperm(corpus_size)].to(
                device
            )  # Extract a smaller corpus
            corpus_latents = classifier.latent_representation(corpus_inputs).detach()
            simplex = Simplex(corpus_inputs, corpus_latents)
            simplex.fit(test_inputs, test_latents, reg_factor=0)
            residuals[id_size, cv] = torch.mean(
                torch.sqrt(
                    torch.sum((simplex.latent_approx() - test_latents) ** 2, dim=-1)
                )
            ).cpu()

    print(residuals.mean(dim=-1))
    print(residuals.std(dim=-1))
################################################################################

def main(experiment: str = "approximation_quality", cv: int = 0) -> None:
    if experiment == "approximation_quality":
        approximation_quality(cv=cv)
    elif experiment == "outlier_detection": # TODO
        outlier_detection(cv=cv)
    elif experiment == "corpus_size": #TODO
        corpus_size_effect()
    else:
        raise ValueError(
            "The name of the experiment is not valid. "
            "Valid names are: approximation_quality , outlier_detection , corpus_size.  "
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-experiment",
        type=str,
        default="outlier_detection",
        help="Experiment to perform",
    )
    parser.add_argument("-cv", type=int, default=0, help="Cross validation parameter")
    args = parser.parse_args()
    main(args.experiment, args.cv)
