import sparse
import json
import numpy as np
import pandas as pd
from collections import Counter

import torch
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split

ROOT_PATH = "/home/mai.kassem/Documents/Projects/abr-ehr/data"
FIDDLE_DATA_PATH = "/l/users/mai.kassem/datasets/preprocessed_fiddle_3"
ICUSTAYS_PATH = f"{FIDDLE_DATA_PATH}/prep/icustays_MV.csv"
FEATURES_PATH = f"{FIDDLE_DATA_PATH}/features/outcome=ARF,T=12.0,dt=1.0/"
LABELS_PATH = f"{FIDDLE_DATA_PATH}/population/ARF_12.0h.csv"
ARF12_PATH = "/l/users/mai.kassem/datasets/preprocessed_fiddle_3/ARF12"


def prepare_mimic_dataset():
    """Prepare the MIMIC dataset for the task of predicting ARF in the next 12 hours."""
    df_icustays_MV = pd.read_csv(ICUSTAYS_PATH).set_index("ICUSTAY_ID")

    # Time independant features
    S = sparse.load_npz(FEATURES_PATH + "S_all.npz")
    S_names = json.load(open(FEATURES_PATH + "S_all.feature_names.json", "r"))
    S_index = pd.read_csv(FEATURES_PATH + "S.ID.csv").set_index(["ID"])
    df_S = pd.DataFrame(S.todense(), columns=S_names, index=S_index.index)

    # Time dependant features
    X = sparse.load_npz(FEATURES_PATH + "X_all.npz")
    X_names = json.load(open(FEATURES_PATH + "X_all.feature_names.json", "r"))
    # X_index = pd.read_csv(FEATURES_PATH + "X.ID,t_range.csv").set_index(["ID", "t_range"])
    X_index = pd.read_csv(FEATURES_PATH + "X.ID,t_range.csv").set_index(["ID"])
    df_X = pd.DataFrame(
        X.todense().reshape(-1, X.shape[-1]), columns=X_names, index=X_index.index
    )

    # Labels
    population_dataset = pd.read_csv(LABELS_PATH).set_index("ID")

    df_merged = pd.merge(
        df_icustays_MV["partition"], df_S, left_index=True, right_index=True
    )
    df_merged = df_merged.set_index(df_S.index)
    df_merged = pd.merge(df_merged, df_X, left_index=True, right_index=True)
    df_merged = df_merged.set_index(df_X.index)
    df_merged = pd.merge(
        df_merged, population_dataset["ARF_LABEL"], left_index=True, right_index=True
    )
    df_merged = df_merged.set_index(df_X.index)
    df_merged = df_merged.drop_duplicates()

    df_train = df_merged[df_merged["partition"] == "train"]
    df_val = df_merged[df_merged["partition"] == "val"]
    df_test = df_merged[df_merged["partition"] == "test"]
    df_train = df_train.drop(columns=["partition"], axis=1)
    df_val = df_val.drop(columns=["partition"], axis=1)
    df_test = df_test.drop(columns=["partition"], axis=1)

    print(np.unique(df_merged["ARF_LABEL"], return_counts=True))
    df_train.to_csv(ARF12_PATH + "/df_train.csv")
    df_val.to_csv(ARF12_PATH + "/df_val.csv")
    df_test.to_csv(ARF12_PATH + "/df_test.csv")


class CustomDataLoader(torch.utils.data.Dataset):
    """This custom data loader will ensure that you have at least one sample from each class
    stored in the class_indices dictionary."""

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.class_indices = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        target = self.targets[index]
        if target not in self.class_indices:
            self.class_indices[target] = index
        return self.data[index], self.targets[index], index


def load_dataset(batch_size=32):
    """Returns training and test sets."""
    df_train = pd.read_csv(ARF12_PATH + "/df_train.csv").set_index("ID")
    df_val = pd.read_csv(ARF12_PATH + "/df_val.csv").set_index("ID")
    df_test = pd.read_csv(ARF12_PATH + "/df_test.csv").set_index("ID")
    print(df_train.head(1))

    X_train = np.array((df_train.drop(columns=["ARF_LABEL"], axis=1)))
    X_val = np.array((df_val.drop(columns=["ARF_LABEL"], axis=1)))
    X_test = np.array((df_test.drop(columns=["ARF_LABEL"], axis=1)))
    y_train = np.array((df_train["ARF_LABEL"]))
    y_val = np.array((df_val["ARF_LABEL"]))
    y_test = np.array((df_test["ARF_LABEL"]))

    print(X_train)
    print(y_train)

    feature_size = X_train.shape[1]  # dimension of input data
    print("Feature size: ", feature_size)
    num_classes = 2  # number of categories
    kwargs = {"num_workers": 0, "pin_memory": False}

    train_set = CustomDataLoader(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )

    val_set = CustomDataLoader(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )

    test_set = CustomDataLoader(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, **kwargs
    )
    print("Train set size: ", train_loader.dataset.data.shape)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True, **kwargs
    )
    print("Val set size: ", val_loader.dataset.data.shape)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, **kwargs
    )
    print("Test set size: ", test_loader.dataset.data.shape)

    # get number of samples per class in train dataset
    classes, class_counts = np.unique(
        np.array(train_loader.dataset.targets), return_counts=True
    )

    print("Number of samples per class: ", class_counts)

    data_splits = {"train": train_set, "val": val_set, "test": test_set}
    data_loaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    print("data type: ", type(train_loader.dataset.data))

    return data_splits, data_loaders, class_counts
