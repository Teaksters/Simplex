import argparse
import os
import pickle as pkl
from pathlib import Path
import numpy as np

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
DATA_DIR = Path('../Data/preprocessed')


class MimicDataset(Dataset):
    def __init__(self, X, y=[]) -> None:
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
    pd.options.mode.chained_assignment = None  # default='warn'
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

def generate_episode_dict(df, col, slice=False, partition=1.0):
    df_col = df[col]
    if partition == 1.0:
        episode_dict = {col + ' mean ' + str(partition): df_col.mean()}
        episode_dict_std = {col + ' std ' + str(partition): df_col.std()}
        episode_dict_min = {col + ' min ' + str(partition): df_col.min()}
        episode_dict_max = {col + ' max ' + str(partition): df_col.max()}

        # MIGHT TAKE THIS OUT JUST IN CASE IT MIGHT BE TOO EASY
        episode_dict_len = {col + ' len ': df_col.size}
        episode_dict.update(episode_dict_len)

    elif slice < 0:
        episode_dict = {col + ' mean ' + str(partition): df_col[slice:].mean()}
        episode_dict_std = {col + ' std ' + str(partition): df_col[slice:].std()}
        episode_dict_min = {col + ' min ' + str(partition): df_col[slice:].min()}
        episode_dict_max = {col + ' max ' + str(partition): df_col[slice:].max()}
    else:
        episode_dict = {col + ' mean ' + str(partition): df_col[:slice].mean()}
        episode_dict_std = {col + ' std ' + str(partition): df_col[:slice].std()}
        episode_dict_min = {col + ' min ' + str(partition): df_col[:slice].min()}
        episode_dict_max = {col + ' max ' + str(partition): df_col[:slice].max()}

    episode_dict.update(episode_dict_std)
    episode_dict.update(episode_dict_min)
    episode_dict.update(episode_dict_max)

    # Replace nan with standard numeric value
    episode_dict = {k: 0.0 if np.isnan(v) else v for k, v in episode_dict.items()}
    return episode_dict

def load_timeseries(): # COULD BE USED FOR MORE VALUES LATER BY NOT DROPPING THOSE COLS
    pd.options.mode.chained_assignment = None  # default='warn'
    desired_cols = ['Diastolic blood pressure',
                    # 'Glascow coma scale total',
                    'Glucose',
                    'Heart Rate',
                    'Height',
                    'Mean blood pressure',
                    'Oxygen saturation',
                    'Respiratory rate',
                    'Systolic blood pressure',
                    'Temperature',
                    'Weight',
                    'pH']

    data_folders = ['train',
                    'test']

    final_df = pd.DataFrame()
    # Generate paths to data
    paths = [DATA_DIR / path for path in data_folders]
    paths = [folder / sample
                for folder in paths
                    for sample in os.listdir(folder)]
    timeserie_paths = [path / file
                for path in paths
                    for file in os.listdir(path)
                        if file[-14:] == 'timeseries.csv']

    # Prepare the subset for episode time serie feature
    sub_sequences = np.array([0.1, 0.25, 0.5])

    print('Reading and gathering timeserie data, this can take a while...')
    for i, path in enumerate(timeserie_paths):
        # Read timeseries data of first 48 hours into dataframe
        episode_df = pd.read_csv(path)
        episode_df = episode_df[episode_df.Hours < 48.0]

        # Prepare the subset slices for episode time serie feature
        pos_slice = (sub_sequences * episode_df.shape[0]).astype(int)

        # Gather and tabularize statistics on timeserie
        stay = '_'.join(path.parts[-2:])
        episode_dict = {}
        episode_dict['stay'] = stay
        for col in desired_cols:
            # For singular measurements have special handling
            if col == 'Weight' or col == 'Height':
                if episode_df[col].notnull().values.any():
                    episode_dict_col = {col: episode_df[col].mean()}
                    episode_dict.update(episode_dict_col)
                else:
                    episode_dict_col = {col: 0.0}
                    episode_dict.update(episode_dict_col)
                continue
            # First the full time serie
            episode_dict_col = generate_episode_dict(episode_df, col)
            episode_dict.update(episode_dict_col)
            for slice, partition in zip(pos_slice, sub_sequences):
                # Then different partitions
                episode_dict_col = generate_episode_dict(episode_df, col, slice, partition)
                episode_dict.update(episode_dict_col)
                episode_dict_col = generate_episode_dict(episode_df, col, -slice, -partition)
                episode_dict.update(episode_dict_col)

        # Create DataFrame
        if i == 0:
            final_df = pd.DataFrame(episode_dict, index=[i])
        # Add new data to existing DataFrame
        else:
            episode_df = pd.DataFrame(episode_dict, index=[i])
            final_df = pd.concat([final_df, episode_df])
    return final_df


def load_tabular_mimic(random_seed: int = 42) -> tuple:
    # Specify undesired columns
    drop_cols = ['stay', 'period_length']
    data_df = 0

    # Prepare data directory
    pkl_dir = DATA_DIR / 'pickles'
    if not pkl_dir.exists():
        print(f"Creating the pickle directory {pkl_dir}")
        os.makedirs(pkl_dir)
    data_pickle = 'tab_mort_data.pkl'
    pickle_path = pkl_dir / data_pickle

    # If data is preprocessed load it
    if pickle_path.exists():
        print('loading preprocessed data!')
        with open(pickle_path, "rb") as f:
            data_df = pkl.load(f)

    # Otherwise preprocess data
    else:
        print('preprocessing data, this only happens once!')
        # Load MIMIC-III data into panda dataframes
        label_dir = os.path.join(DATA_DIR, 'in-hospital-mortality')
        label_df = load_from_preprocessed(label_dir)
        feature_dir = os.path.join(DATA_DIR, 'phenotyping')
        feature_df = load_from_preprocessed(feature_dir)
        age_df = load_age()
        time_df = load_timeseries()

        # Merge data into workable complete format
        data_df = pd.merge(label_df, feature_df, on='stay')
        data_df = pd.merge(time_df, data_df, on='stay')
        data_df = pd.merge(age_df, data_df, on='stay')
        data_df.drop(columns=drop_cols, inplace=True)

        # Safe pickle for later use
        print('Done, storing data for quick retrieval next time.')
        with open(pickle_path, "wb") as f:
            pkl.dump(data_df, f)

    df = sklearn.utils.shuffle(data_df, random_state=random_seed)
    df = df.reset_index(drop=True)


    features, labels = df.loc[:, df.columns != 'y_true'], df['y_true']
    return features, labels

def approximation_quality(
    cv: int = 0,
    age_scaler: float=1.,
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
    corpus_latent_reps = classifier.latent_representation(corpus_data).detach()
    corpus_probas = classifier.probabilities(corpus_data).detach()
    corpus_true_classes = torch.zeros(corpus_probas.shape, device=device)
    corpus_true_classes[
        torch.arange(corpus_size), corpus_target.type(torch.LongTensor)
    ] = 1
    test_latent_reps = classifier.latent_representation(test_data).detach()

    ############################################################################

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


def main(experiment: str = "approximation_quality", cv: int = 0, age_scaler: float = 1.) -> None:
    if experiment == "approximation_quality":
        approximation_quality(cv=cv, age_scaler=age_scaler)
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
        default="approximation_quality",
        help="Experiment to perform",
    )
    parser.add_argument("-cv", type=int, default=0, help="Cross validation parameter")
    parser.add_argument("-age_scaler", type=float, default=1., help="Scaling variable for sample ages")
    args = parser.parse_args()
    main(args.experiment, args.cv, args.age_scaler)
