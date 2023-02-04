import sparse
import json
import numpy as np
import pandas as pd
from collections import Counter

import torch
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split

ROOT_PATH = "/home/mai.kassem/Documents/Projects/abr-ehr/data"
ARF12_PATH = f"{ROOT_PATH}/preprocessed_fiddle_3/features/outcome=ARF,T=12.0,dt=1.0/"
ICUSTAYS_PATH = f"{ROOT_PATH}/preprocessed_fiddle_3/prep/icustays_MV.csv"
LABELS_PATH = f"{ROOT_PATH}/preprocessed_fiddle_3/population/ARF_12.0h.csv"


def prepare_mimic_dataset():
    df_icustays_MV = pd.read_csv(ICUSTAYS_PATH).set_index("ICUSTAY_ID")

    # Time independant features
    S = sparse.load_npz(ARF12_PATH + "S_all.npz")
    S_names = json.load(open(ARF12_PATH + "S_all.feature_names.json", "r"))
    S_index = pd.read_csv(ARF12_PATH + "S.ID.csv").set_index(["ID"])
    df_S = pd.DataFrame(S.todense(), columns=S_names, index=S_index.index)

    # Time dependant features
    X = sparse.load_npz(ARF12_PATH + "X_all.npz")
    X_names = json.load(open(ARF12_PATH + "X_all.feature_names.json", "r"))
    # X_index = pd.read_csv(ARF12_PATH + "X.ID,t_range.csv").set_index(["ID", "t_range"])
    X_index = pd.read_csv(ARF12_PATH + "X.ID,t_range.csv").set_index(["ID"])
    df_X = pd.DataFrame(
        X.todense().reshape(-1, X.shape[-1]), columns=X_names, index=X_index.index
    )

    # Labels
    population_dataset = pd.read_csv(LABELS_PATH).set_index("ID")

    df_merged = pd.merge(
        df_icustays_MV["partition"],
        df_S,
        left_index=True,
        right_index=True,
        how="inner",
    )
    df_merged = pd.merge(
        df_merged, df_X, left_index=True, right_index=True, how="inner"
    )
    df_merged = pd.merge(
        population_dataset["ARF_LABEL"],
        df_merged,
        left_index=True,
        right_index=True,
        how="inner",
    )
    df_train = df_merged[df_merged["partition"] == "train"]
    df_val = df_merged[df_merged["partition"] == "val"]
    df_test = df_merged[df_merged["partition"] == "test"]
    df_train = df_train.drop(columns=["partition"], axis=1)
    df_val = df_val.drop(columns=["partition"], axis=1)
    df_test = df_test.drop(columns=["partition"], axis=1)

    print(np.unique(df_merged["ARF_LABEL"], return_counts=True))
    df_train.to_csv(ROOT_PATH + "/df_train.csv")
    df_val.to_csv(ROOT_PATH + "/df_val.csv")
    df_test.to_csv(ROOT_PATH + "/df_test.csv")


# create custom dataset torch
class MyCustomDataset(Dataset):
    def __init__(self, X, Y):
        self.data = X
        self.targets = Y

    def __getitem__(self, index):
        return (self.data[index], self.targets[index], index)

    def __len__(self):
        return len(self.data)


def load_dataset(batch_size=32):
    df_train = pd.read_csv(ROOT_PATH + "/df_train.csv")
    df_val = pd.read_csv(ROOT_PATH + "/df_val.csv")
    df_test = pd.read_csv(ROOT_PATH + "/df_test.csv")
    df_train = df_train.drop(columns=["Unnamed: 0"], axis=1)
    df_val = df_val.drop(columns=["Unnamed: 0"], axis=1)
    df_test = df_test.drop(columns=["Unnamed: 0"], axis=1)
    print(df_train.head(1))

    X_train = np.array(df_train.drop(columns=["ARF_LABEL"], axis=1))
    X_val = np.array(df_val.drop(columns=["ARF_LABEL"], axis=1))
    X_test = np.array(df_test.drop(columns=["ARF_LABEL"], axis=1))
    y_train = np.array(df_train["ARF_LABEL"])
    y_val = np.array(df_val["ARF_LABEL"])
    y_test = np.array(df_test["ARF_LABEL"])

    print(X_train)
    print(y_train)

    feature_size = X_train.shape[1]  # dimension of input data
    print("Feature size: ", feature_size)
    num_classes = 2  # number of categories
    kwargs = {"num_workers": 0, "pin_memory": False}

    train_set = MyCustomDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )

    val_set = MyCustomDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )

    test_set = MyCustomDataset(
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


def load_features():
    # TODO train, test.. icustays_MV
    # TODO time dep.
    # Time invariant features
    S = sparse.load_npz(ARF12_PATH + "S_all.npz")
    S_names = json.load(open(ARF12_PATH + "S_all.feature_names.json", "r"))
    S_index = pd.read_csv(ARF12_PATH + "S.ID.csv").set_index(["ID"])
    df_S = pd.DataFrame(S.todense(), columns=S_names, index=S_index.index)

    # Time dependant features
    # X = sparse.load_npz(ARF12_PATH + "X_all.npz")
    # X_names = json.load(open(ARF12_PATH + "X_all.feature_names.json", "r"))
    # X_index = pd.read_csv(ARF12_PATH + "X.ID,t_range.csv").set_index(["ID", "t_range"])
    # df_X = pd.DataFrame(
    #     X.todense().reshape(-1, X.shape[-1]), columns=X_names, index=X_index.index
    # )

    # display(df_S)
    # display(df_X)
    return S.todense()


def load_labels():
    population_dataset = pd.read_csv(
        ROOT_PATH + "/preprocessed_fiddle_3/population/ARF_12.0h.csv"
    )
    return np.array(population_dataset["ARF_LABEL"])


def load_fiddle_dataset(batch_size=32):
    X = load_features()
    Y = load_labels()
    feature_size = X.shape[1]  # dimension of input data
    num_classes = 2  # number of categories
    kwargs = {"num_workers": 0, "pin_memory": False}

    # split dataset into train, val, test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    print(X_train)

    # train_set = torch.utils.data.TensorDataset(
    #     torch.from_numpy(X_train.astype(float)), torch.from_numpy(y_train)
    # )
    train_set = MyCustomDataset(
        torch.from_numpy(X_train.astype(float)), torch.from_numpy(y_train)
    )

    # val_set = torch.utils.data.TensorDataset(
    #     torch.from_numpy(X_val.astype(float)), torch.from_numpy(y_val)
    # )
    val_set = MyCustomDataset(
        torch.from_numpy(X_val.astype(float)), torch.from_numpy(y_val)
    )

    # test_set = torch.utils.data.TensorDataset(
    #     torch.from_numpy(X_test.astype(float)), torch.from_numpy(y_test)
    # )
    test_set = MyCustomDataset(
        torch.from_numpy(X_test.astype(float)), torch.from_numpy(y_test)
    )

    # print("Dataset size: ", len(dataset))
    # train_set, val_set, test_set = torch.utils.data.random_split(
    #     dataset, [len(Y) * 0.8, len(Y) * 0.1, len(Y) * 0.1]
    # )
    # print("Train set size: ", len(train_set))

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
