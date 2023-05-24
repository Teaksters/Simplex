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

from simplexai.experiments.tabular_mimic import MimicDataset, \
                                                load_from_preprocessed, \
                                                load_age, \
                                                load_tabular_mimic


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = '../Data/preprocessed'




def main():
    # Load the data
    X_df, y_df = load_tabular_mimic()
    diagnoses_df = X_df.iloc[:, -23:]

    # Interpret grand scale patterns
    diagnoses = list(diagnoses_df)
    occurances = diagnoses_df.sum(axis=0)
    diagnoses_probs = occurances / occurances.sum()

    # Generate diagnoses prototypes using averaging and thresholding
    prototype_df = pd.DataFrame(columns=list(X_df))
    print(prototype_df)
    for diagnosis in diagnoses:
        single_df = X_df.loc[X_df[diagnosis] == 1.]
        prototype = single_df.mean(axis=0)
        print(single_df)
        prototype_df.append(prototype)
        # prototype_df = pd.concat([prototype_df, single_df], ignore_index=True)
    print(prototype_df)
    return 0


if __name__=='__main__':
    main()
