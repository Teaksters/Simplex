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
import copy

from simplexai.experiments.tabular_mimic import MimicDataset, \
                                                load_from_preprocessed, \
                                                load_age, \
                                                load_tabular_mimic


DATA_DIR = Path('../Data/preprocessed')


def create_corpus(data, n_bin=25):
    # Replace all place holders for nan with nan again for later calculations
    data.iloc[:, :-n_bin][data.iloc[:, :-n_bin] <= 0] = np.nan
    diagnoses_df = data.iloc[:, -n_bin:]

    # Calculate occurance probabilities (thresholds)
    diagnoses = list(diagnoses_df)
    occurances = diagnoses_df.sum()
    diagnoses_probs = occurances / occurances.sum()

    # Generate diagnoses prototypes using mean values for each diagnosis
    prototype_df = []
    for diagnosis in diagnoses:
        single_df = data.loc[data[diagnosis] == 1.]
        prototype = single_df.mean()

        # Normalize binary diagnoses features into probability function
        prototype.iloc[-n_bin:] /= prototype.iloc[-n_bin:].sum()
        prototype = dict(prototype)
        prototype_df.append(prototype)

    # Binarize diagnosis features using the occurance probability as threshold
    prototype_df = pd.DataFrame.from_dict(prototype_df)
    prototype_df.iloc[:, -n_bin:] = (prototype_df.iloc[:, -n_bin:] >= diagnoses_probs).astype('float')
    return prototype_df

def main():
    # Load the data
    X_df, _ = load_tabular_mimic()
    prototypes = create_corpus(X_df)

    for col in prototypes:
        print(prototypes[col])
    print(prototypes)

    # Store the created corpus
    corpus_path = DATA_DIR / 'pickles' / 'tab_mort_diagnosis_prototypes.pkl'
    print('Done, storing corpus for later.')
    with open(corpus_path, "wb") as f:
        pkl.dump(prototypes, f)
    return 0


if __name__=='__main__':
    main()
