import argparse
import pickle as pkl
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import torch

from simplexai.models.tabular_data import MortalityPredictor


current_path = Path.cwd()
load_path = current_path / "experiments/results/mimic/quality/tuning/epochs"

train_losses = []
test_losses = []
test_accs = []
epochs = []
for epoch_path in os.listdir(load_path):
    epochs.append(int(epoch_path))
    temp_path = load_path / epoch_path
    for data_path in os.listdir(temp_path):
        if data_path[-4:] == '.pkl':
            temp_path = temp_path / data_path
            file = open(temp_path, 'rb')
            print(temp_path)
            data = pkl.load(file)
            train_losses.append(data[0])
            test_losses.append(data[2])
            test_accs.append(data[3])

train_losses = np.array(train_losses)
test_losses = np.array(test_losses)
test_accs = np.array(test_accs)

print('train_losses:\n', train_losses)
print('test_losses:\n', test_losses)
print('test_accs:\n', test_accs)
