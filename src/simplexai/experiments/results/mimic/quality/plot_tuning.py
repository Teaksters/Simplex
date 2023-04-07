import argparse
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import torch

from simplexai.models.tabular_data import MortalityPredictor

load_path = current_path / "experiments/results/mimic/quality/tuning/epochs"

for epoch_path in os.listdir(load_path):
    temp_path = load_path / epoch_path
    for data_path in os.listdir(temp_path):
        if data[-4:] == '.pkl':
            temp_path = temp_path / data_path
            file = open(temp_path, 'rb')
            data = pkl.load(file)
            print(epoch_path, data_path, data)
            exit()
