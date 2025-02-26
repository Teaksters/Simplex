import argparse
import pickle as pkl
from pathlib import Path
import os
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import torch

from simplexai.models.tabular_data import MortalityPredictor


class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


current_path = Path.cwd()
load_path = current_path / "experiments/results/mimic/quality/tuning/epochs"

train_losses = []
val_losses = []
val_AUCs = []
train_counter = []
epochs = []
test_AUC = []
test_acc = []
for epoch_path in os.listdir(load_path):
    epochs.append(int(epoch_path))
    cur_path = load_path / epoch_path
    for data_path in os.listdir(cur_path):
        if epoch_path == '20':
            if data_path[-4:] == '.pkl' and data_path[:4] != 'test':
                temp_path = cur_path / data_path
                file = open(temp_path, 'rb')
                data = CPU_Unpickler(file).load()
                train_losses.append(data[0])
                train_counter = data[1]
                val_losses.append(data[2])
                val_AUCs.append(data[-1])
        if epoch_path == '5':
            if data_path[-4:] == '.pkl' and data_path[:4] == 'test':
                temp_path = cur_path / data_path
                file = open(temp_path, 'rb')
                data = CPU_Unpickler(file).load()
                test_acc.append(data[0][0].item())
                test_AUC.append(data[1])

# train_losses = np.array(train_losses)
# print(train_losses)
val_losses = np.array(val_losses)
val_AUCs = np.array(val_AUCs)
test_acc = np.array(test_acc)
test_AUC = np.array(test_AUC)

# train_mean = np.mean(train_losses, axis=0)
# train_std = train_losses.std(axis=0)
val_mean = val_losses.mean(axis=0)
val_std = val_losses.std(axis=0)

# print('train_losses:\n', train_mean, train_std)
print('val_losses:\n', val_mean, val_std)
print('test_performance:\n', test_acc.mean(), "(+-", test_acc.std(), ")", "(accuracy)", test_AUC.mean(), "(+-", test_AUC.std(), ")", '(AUC)')

plt.figure(1)
plt.plot(train_counter, train_mean, label='train loss')
plt.fill_between(train_mean - train_std, train_mean + train_std)
plt.plot(test_mean, label='test loss')
plt.fill_between(test_mean - test_std, test_mean + test_std)
plt.legend()
plt.tight_layout()
plt.savefig('experiments/results/mimic/quality/tuning/plots/epoch_fitting.png')
