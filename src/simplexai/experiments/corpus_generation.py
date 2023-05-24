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

    # Organize it
    diagnoses = list(diagnoses_df)
    occurances = diagnoses_df.sum(axis=0)
    diagnoses_probs = occurances / occurances.sum()
    print(diagnoses_df)
    print(occurances, diagnoses, diagnoses_probs, diagnoses_probs.sum())
    return 0


if __name__=='__main__':
    main()
