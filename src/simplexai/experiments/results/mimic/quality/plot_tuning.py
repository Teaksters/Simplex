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
test_AUC = 0
test_acc = 0
for epoch_path in os.listdir(load_path):
    if epoch_path != '20': continue
    epochs.append(int(epoch_path))
    cur_path = load_path / epoch_path
    for data_path in os.listdir(cur_path):
        if data_path[-4:] == '.pkl' and data_path[:4] != 'test':
            temp_path = cur_path / data_path
            file = open(temp_path, 'rb')
            data = CPU_Unpickler(file).load()
            print(len(data[0]))
            train_losses.append(data[0])
            train_counter = data[1]
            val_losses.append(data[2])
            val_AUCs.append(data[-1])
        elif data_path[-4:] == '.pkl':
            test_acc = data[0][0]
            test_AUC = data[1]

# train_losses = np.array(train_losses)
# print(train_losses)
val_losses = np.array(val_losses)
val_AUCs = np.array(val_AUCs)

# train_mean = np.mean(train_losses, axis=0)
# train_std = train_losses.std(axis=0)
val_mean = val_losses.mean(axis=0)
val_std = val_losses.std(axis=0)

print('xxxxxxxxxxxxxxxxxxEPOCH: ', epoch_path, 'xxxxxxxxxxxxxxx')
print('train_losses:\n', train_mean, train_std)
print('val_losses:\n', val_mean, val_std)
print('val_AUCs:\n', test_AUCs.mean(), "+- (", test_AUCs.std(), ')')
print('test_performance:\n', test_acc, "(accuracy)", test_AUC, '(AUC)')

plt.figure(1)
plt.plot(train_counter, train_mean, label='train loss')
plt.fill_between(train_mean - train_std, train_mean + train_std)
plt.plot(test_mean, label='test loss')
plt.fill_between(test_mean - test_std, test_mean + test_std)
plt.legend()
plt.tight_layout()
plt.savefig('experiments/results/mimic/quality/tuning/plots/epoch_fitting.png')
