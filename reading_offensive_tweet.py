"""
Reading the "offensive" dataset.
"""

from os.path import join

import numpy as np
import pandas as pd


DATASET_HOME = "./datasets/offensive/"


def read_label(filename: str) -> list:
    with open(join(DATASET_HOME, filename)) as f:
        labels = f.readlines()
    labels = [int(i.strip()) for i in labels]
    return labels


def read_text(filename: str) -> list:
    with open(join(DATASET_HOME, filename)) as f:
        labels = f.readlines()
    labels = [i.strip() for i in labels]
    return labels


def read_mapping(filename: str) -> dict:
    with open(join(DATASET_HOME, filename), 'r') as f:
        mappings = f.readlines()
    mappings = [i.strip().split('\t') for i in mappings]
    mappings = {int(m[0]): m[1] for m in mappings}
    return mappings


def get_offensive_data():

    files = ["test_labels.txt", "test_text.txt",
             "train_labels.txt", "train_text.txt",
             "val_labels.txt", "val_text.txt"]
    mappings = read_mapping("mapping.txt")

    for i in files:
        if "labels" in i:
            if "test" in i:
                test_labels = np.array(read_label(i))
            elif "train" in i:
                train_labels = np.array(read_label(i))
            elif "val" in i:
                val_label = np.array(read_label(i))
        elif "text" in i:
            if "test" in i:
                test_text = read_text(i)
            elif "train" in i:
                train_text = read_text(i)
            elif "val" in i:
                val_text = read_text(i)

    test_data = {'tweets': test_text, 'labels': test_labels}
    test_data = pd.DataFrame.from_dict(test_data)
    del test_labels, test_text

    train_data = {'tweets': train_text, 'labels': train_labels}
    train_data = pd.DataFrame.from_dict(train_data)
    del train_labels, train_text

    val_data = {'tweets': val_text, 'labels': val_label}
    val_data = pd.DataFrame.from_dict(val_data)
    del val_label, val_text

    return train_data, test_data, val_data, mappings

# print("Shape of test dataset {}".format(test_data.shape))
# print("Shape of training dataset {}".format(train_data.shape))
# print("Shape of validation dataset {}".format(val_data.shape))
# print("Some of the datapoints : \n")
# print(train_data.head())
# print("Mapping of labels : {}".format(mappings))
