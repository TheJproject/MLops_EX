import torch
import numpy as np
import glob
from pathlib import Path
from torch.utils.data import Dataset


def mnist():
    # exchange with the corrupted mnist dataset
    data_folder = str(Path().resolve()) + '\\data\\corruptmnist'
    file_list = glob.glob(data_folder + r'/train/*')
    data_all = [np.load(fname) for fname in file_list]
    train_all = {}
    for data in data_all:
        [train_all.update({k: v}) for k, v in data.items()]
    np.savez(data_folder + '\\train\\train_merged.npz', **train_all)
    test = np.load(data_folder + '\\test\\test.npz')
    return train_all, test

