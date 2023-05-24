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


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = '../Data/preprocessed'




def main():
    # Load the data
    X_df, y_df = load_tabular_mimic()
    # Replace all place holders for nan with nan again for later calculations
    X_df.iloc[:, :-25][X_df.iloc[:, :-25] <= 0] = np.nan

    diagnoses_df = X_df.iloc[:, -25:]

    # Interpret grand scale patterns
    diagnoses = list(diagnoses_df)
    occurances = diagnoses_df.sum()
    diagnoses_probs = occurances / occurances.sum()

    # Generate diagnoses prototypes using mean values for each diagnosis
    prototype_df = []
    for diagnosis in diagnoses:
        single_df = X_df.loc[X_df[diagnosis] == 1.]
        prototype = single_df.mean()

        ######################### SO FAR SO GOOD #####################3
        # Normalize binary diagnoses features into probability function
        prototype.iloc[-25:] /= prototype.iloc[-25:].sum()
        prototype = dict(prototype)
        prototype_df.append(prototype)

    # Binarize diagnosis features using the occurance probability as threshold
    prototype_df = pd.DataFrame.from_dict(prototype_df)
    print(prototype_df.iloc[0, -25:], '\n', diagnoses_probs)
    prototype_df.iloc[:, -25:] = (prototype_df.iloc[:, -25:] >= diagnoses_probs).astype('float')
    # for row in prototype_df:
    #     print(prototype_df[row])
    return 0


if __name__=='__main__':
    main()
