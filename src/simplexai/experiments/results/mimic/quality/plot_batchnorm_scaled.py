import argparse
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import torch
import os
import io


class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

################## WORKING ########################################

parser = argparse.ArgumentParser()
parser.add_argument(
    "-cv_list",
    nargs="+",
    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    help="The list of experiment cv identifiers to plot",
    type=int,
)
parser.add_argument(
    "-age_scalers",
    nargs='*',
    type=float,
    default=[1.0, 2.0, 3.0, 4.0, 5.0],
    help="Scaling variable for sample ages")

args = parser.parse_args()
cv_list = args.cv_list
scalers = args.age_scalers

current_path = Path.cwd()
load_path = current_path / "experiments/results/mimic/quality/scaled/"

for scaler in scalers:
    for cv in cv_list:
        current_path = load_path / str(scaler) / f"BNorm_out_data_cv{cv}.pkl"
        with open(current_path, rb) as f:
            data = CPU_Unpickler(file).load()
        print(data)
        # TODO: load the bathnorm output data
        # measure them somehow

# TODO: plot the metrics (in the correct directory not where I am loading from rn)
safe_path = load_path / 'plots/'
if not os.path.exists(safe_path):
    os.makedirs(safe_path)
