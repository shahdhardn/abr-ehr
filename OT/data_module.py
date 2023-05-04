import re
from typing import Any
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from torch.utils.data.sampler import WeightedRandomSampler
import pytorch_lightning as pl
import pandas as pd
import sparse
from transformers import BertTokenizer
from pandarallel import pandarallel
import h5py
import copy


class LabeledSubset(Subset):
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, y, index


# class EHRDataset(Dataset):
#     def __init__(self, split, notes, discrete, merge):
#         self.class_indices = {}
#         self.notes = notes
#         self.discrete = discrete
#         self.merge = merge
#         self.X = split["X"][()]
#         self.s = split["S"][()]
#         self.y = split["label"][()]
#         assert len(self.X) == len(self.s) and len(self.X) == len(self.y)
#         if self.notes:
#             if self.discrete:
#                 self.time = split["time"][()]
#             self.input_ids = split["input_ids"][()]
#             self.token_type_ids = split["token_type_ids"][()]
#             self.attention_mask = split["attention_mask"][()]
#             assert len(self.input_ids) == len(self.y)

#     def __getitem__(self, index):
#         xi = self.X[index]
#         si = self.s[index]
#         L, D = xi.shape
#         if self.merge:
#             xi = np.hstack((xi, np.tile(si, (L, 1))))  # time dependent
#             x = torch.from_numpy(xi).float()
#         else:
#             si = torch.from_numpy(si).float()  # time invariant
#             xi = torch.from_numpy(xi).float()
#             x = (si, xi)
#         if self.notes:
#             if self.discrete:
#                 base = torch.zeros((L, self.input_ids[0].shape[-1]))
#                 input_ids = torch.scatter(
#                     base, 0, self.time[index], self.input_ids[index]
#                 )
#                 token_type_ids = torch.scatter(
#                     base, 0, self.time[index], self.token_type_ids[index]
#                 )
#                 attention_mask = torch.scatter(
#                     base, 0, self.time[index], self.attention_mask[index]
#                 )
#             else:
#                 input_ids = torch.tensor(self.input_ids[index])
#                 token_type_ids = torch.tensor(self.token_type_ids[index])
#                 attention_mask = torch.tensor(self.attention_mask[index])
#             x = (x, input_ids, token_type_ids, attention_mask)
#         # y = torch.tensor(self.y[index]).float()
#         target = self.y[index]
#         if target not in self.class_indices:
#             self.class_indices[target] = index
#         y = torch.LongTensor([np.int64(self.y[index])])
#         return x, y, index

#     def __len__(self):
#         return len(self.y)


class EHRDataset(Dataset):
    def __init__(self, split, notes, discrete, merge):
        self.notes = notes
        self.discrete = discrete
        self.merge = merge
        self.X = split["X"][()]
        self.s = split["S"][()]
        self.y = split["label"][()]
        assert len(self.X) == len(self.s) and len(self.X) == len(self.y)
        if self.notes:
            if self.discrete:
                self.time = split["time"][()]
            self.input_ids = split["input_ids"][()]
            self.token_type_ids = split["token_type_ids"][()]
            self.attention_mask = split["attention_mask"][()]
            assert len(self.input_ids) == len(self.y)

    def __getitem__(self, index):
        xi = self.X[index]
        si = self.s[index]
        L, D = xi.shape
        if self.merge:
            xi = np.hstack((xi, np.tile(si, (L, 1))))  # time dependent
            x = torch.from_numpy(xi).float()
        else:
            si = torch.from_numpy(si).float()  # time invariant
            xi = torch.from_numpy(xi).float()
            x = (si, xi)
        if self.notes:
            if self.discrete:
                base = torch.zeros((L, self.input_ids[0].shape[-1]))
                input_ids = torch.scatter(
                    base, 0, self.time[index], self.input_ids[index]
                )
                token_type_ids = torch.scatter(
                    base, 0, self.time[index], self.token_type_ids[index]
                )
                attention_mask = torch.scatter(
                    base, 0, self.time[index], self.attention_mask[index]
                )
            else:
                input_ids = torch.tensor(self.input_ids[index])
                token_type_ids = torch.tensor(self.token_type_ids[index])
                attention_mask = torch.tensor(self.attention_mask[index])
            x = (x, input_ids, token_type_ids, attention_mask)
            y = torch.LongTensor([np.int64(self.y[index])])
        return x, y, index

    def __len__(self):
        return len(self.y)


class MimicDataModule(pl.LightningDataModule):
    def __init__(
        self,
        #  mimic_dir: str = '/home/shahad.hardan/Documents/ABR_Research/physionet.org/files/mimiciii/1.4/',
        #  data_dir: str = '/home/shahad.hardan/Documents/research_tf/preprocessed_fiddle_3/',
        mimic_dir: str = "/l/users/mai.kassem/datasets/physionet.org/files/mimiciii/1.4/",
        data_dir: str = "/l/users/mai.kassem/datasets/preprocessed_fiddle_3/",
        train=None,
        val=None,
        meta=None,
        test=None,
        task: str = "ARF",
        duration: float = 12.0,
        timestep: float = 1.0,
        batch_size: int = 40,
        notes: bool = True,
        discrete: bool = False,
        merge: bool = False,
        num_workers: int = 12,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.mimic_dir = Path(mimic_dir)
        self.data_dir = Path(data_dir)
        self.train = train
        self.val = val
        self.meta = meta
        self.test = test
        self.task = task
        self.duration = duration
        self.timestep = timestep
        self.batch_size = batch_size
        self.label_dir = self.data_dir / f"population/{task}_{duration}h.csv"
        # self.label_name = f'{task}_{duration}h'
        self.subject_dir = self.data_dir / "prep/icustays_MV.csv"
        self.task_dir = (
            self.data_dir / f"features/outcome={task},T={duration},dt={timestep}"
        )
        self.X_dir = self.task_dir / "X.npz"
        self.s_dir = self.task_dir / "S.npz"
        self.notes = notes
        self.discrete = discrete
        self.merge = merge
        self.num_workers = num_workers

    def xs_hdf5(self):
        if not Path.is_file(self.task_dir / "Xs.hdf5"):
            X = sparse.load_npz(self.X_dir).todense()
            s = sparse.load_npz(self.s_dir).todense()
            with h5py.File(self.task_dir / "Xs.hdf5", "w") as hf:
                hf.create_dataset("X", data=X)
                hf.create_dataset("S", data=s)

    @staticmethod
    def tokenize(id_texts):
        pandarallel.initialize(nb_workers=12)

        def preprocess1(x):
            y = re.sub("\\[(.*?)]", "", x)
            y = re.sub("[0-9]+\.", "", y)
            y = re.sub("dr\.", "doctor", y)
            y = re.sub("m\.d\.", "md", y)
            y = re.sub("admission date:", "", y)
            y = re.sub("discharge date:", "", y)
            y = re.sub("--|__|==", "", y)
            return y

        def preprocessing(df_less_n):
            df_less_n["TEXT"] = df_less_n["TEXT"].fillna(" ")
            df_less_n["TEXT"] = df_less_n["TEXT"].str.replace("\n", " ")
            df_less_n["TEXT"] = df_less_n["TEXT"].str.replace("\r", " ")
            df_less_n["TEXT"] = df_less_n["TEXT"].apply(str.strip)
            df_less_n["TEXT"] = df_less_n["TEXT"].str.lower()

            df_less_n["TEXT"] = df_less_n["TEXT"].apply(lambda x: preprocess1(x))
            return df_less_n

        id_texts = preprocessing(id_texts)

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        id_texts["FEATS"] = id_texts["TEXT"].parallel_apply(
            lambda x: tokenizer.encode_plus(
                x,
                padding="max_length",
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
            )
        )

        id_texts = id_texts.drop(columns="TEXT")

        id_texts["input_ids"] = id_texts["FEATS"].apply(lambda x: x["input_ids"])

        id_texts["token_type_ids"] = id_texts["FEATS"].apply(
            lambda x: x["token_type_ids"]
        )

        id_texts["attention_mask"] = id_texts["FEATS"].apply(
            lambda x: x["attention_mask"]
        )

        id_texts = id_texts.drop(columns="FEATS")
        return id_texts

    def note_feats(self):
        labels = pd.read_csv(self.label_dir).rename(
            columns={"ID": "ICUSTAY_ID", f"{self.task}_LABEL": "LABEL"}
        )

        notes = pd.read_csv(
            self.mimic_dir / "NOTEEVENTS.csv.gz",
            parse_dates=["CHARTDATE", "CHARTTIME", "STORETIME"],
            compression="gzip",
        )

        icus = (
            pd.read_csv(
                self.mimic_dir / "ICUSTAYS.csv.gz",
                parse_dates=["INTIME", "OUTTIME"],
                compression="gzip",
            )
            .sort_values(by=["SUBJECT_ID"])
            .reset_index(drop=True)
        )

        df = pd.merge(
            icus[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME"]],
            notes,
            on=["SUBJECT_ID", "HADM_ID"],
            how="inner",
        )
        df = df.drop(columns=["SUBJECT_ID", "HADM_ID"])

        df = pd.merge(labels["ICUSTAY_ID"], df, on="ICUSTAY_ID", how="left")

        df = df[df["ISERROR"].isnull()]

        df = df[df["CHARTTIME"].notnull()]

        df["TIME"] = (df["CHARTTIME"] - df["INTIME"]).apply(
            lambda x: x.total_seconds()
        ) / 3600
        df = df[(df["TIME"] <= self.duration) & (df["TIME"] >= 0.0)]
        if self.discrete:
            df["TIME"] = df["TIME"].map(int)
            id_texts = (
                df.groupby(["ICUSTAY_ID", "TIME"])["TEXT"]
                .apply(lambda x: "\n".join(x))
                .reset_index()
            )
            id_texts = self.tokenize(id_texts)

            def func(x, length, dim):
                def stack(name):
                    src = np.zeros((length, dim), dtype=np.int64)
                    idx = x["TIME"].to_numpy()
                    arr = np.stack(x[name].to_numpy())
                    src[idx] = arr
                    return src

                return pd.Series(
                    {
                        "TIME": x["TIME"].to_numpy(),
                        "input_ids": stack("input_ids"),
                        "token_type_ids": stack("token_type_ids"),
                        "attention_mask": stack("attention_mask"),
                    }
                )

            id_texts = id_texts.groupby("ICUSTAY_ID")[
                ["TIME", "input_ids", "token_type_ids", "attention_mask"]
            ].parallel_apply(func, length=int(self.duration), dim=512)
            id_texts = id_texts.drop(columns="TIME").reset_index()
            return id_texts
        else:
            df["VARNAME"] = df[["CATEGORY", "DESCRIPTION"]].apply(
                lambda x: x.CATEGORY + "/" + x.DESCRIPTION, axis=1
            )
            df = df.groupby(["ICUSTAY_ID", "VARNAME"])[["TIME", "TEXT"]].apply(
                lambda x: x.iloc[x["TIME"].argmax()]
            )

            id_texts = (
                df.groupby("ICUSTAY_ID")["TEXT"]
                .apply(lambda x: "\n".join(x))
                .reset_index()
            )
            return self.tokenize(id_texts)

    def note_hdf5(self):
        if self.discrete:
            if not Path.is_file(self.task_dir / "discrete_notes.hdf5"):
                discrete_feats = self.note_feats()
                discrete_feats["ICUSTAY_ID"].to_hdf(
                    self.task_dir / "discrete_notes.hdf5", "ICUSTAY_ID"
                )
                discrete_feats["input_ids"].to_hdf(
                    self.task_dir / "discrete_notes.hdf5", "input_ids"
                )
                discrete_feats["token_type_ids"].to_hdf(
                    self.task_dir / "discrete_notes.hdf5", "token_type_ids"
                )
                discrete_feats["attention_mask"].to_hdf(
                    self.task_dir / "discrete_notes.hdf5", "attention_mask"
                )
        else:
            if not Path.is_file(self.task_dir / "notes.hdf5"):
                note_feats = self.note_feats()
                note_feats.to_hdf(self.task_dir / "notes.hdf5", "notes")

    def save_data(self, file, name, splits, index, xs, df_label):
        hf = h5py.File(file, "a")
        group = hf.create_group(name)
        split_idx = zip(splits, index)
        for s, idx in split_idx:
            s = group.create_group(s)
            s.create_dataset("X", data=xs["X"][idx])
            s.create_dataset("S", data=xs["S"][idx])
            if self.notes:
                if self.discrete:
                    s.create_dataset(
                        "time", data=np.stack(df_label.loc[idx]["TIME"].to_numpy())
                    )
                s.create_dataset(
                    "input_ids",
                    data=np.stack(df_label.loc[idx]["input_ids"].to_numpy()),
                )
                s.create_dataset(
                    "token_type_ids",
                    data=np.stack(df_label.loc[idx]["token_type_ids"].to_numpy()),
                )
                s.create_dataset(
                    "attention_mask",
                    data=np.stack(df_label.loc[idx]["attention_mask"].to_numpy()),
                )
            s.create_dataset(
                "label", data=np.stack(df_label.loc[idx]["LABEL"].to_numpy())
            )

    def split(self, file, k):
        df_label = pd.read_csv(self.label_dir).rename(
            columns={"ID": "ICUSTAY_ID", f"{self.task}_LABEL": "LABEL"}
        )
        if self.notes:
            if self.discrete:
                text_feats = pd.read_hdf(self.task_dir / "notes.hdf5", "discrete_notes")
            else:
                text_feats = pd.read_hdf(self.task_dir / "notes.hdf5", "notes")

            df_label = pd.merge(df_label, text_feats, on="ICUSTAY_ID", how="left")

            df_label = df_label[df_label["input_ids"].notnull()]

        df_subjects = (
            df_label.reset_index()
            .merge(pd.read_csv(self.subject_dir), on="ICUSTAY_ID", how="left")
            .set_index("index")
        )

        train_idx = df_subjects[df_subjects["partition"] == "train"].index.values
        val_idx = df_subjects[df_subjects["partition"] == "val"].index.values
        test_idx = df_subjects[df_subjects["partition"] == "test"].index.values
        xs = h5py.File(self.task_dir / "Xs.hdf5", "r")
        partitions = ["train", "val", "test"]
        indices = [train_idx, val_idx, test_idx]
        self.save_data(file, k, partitions, indices, xs, df_label)
        xs.close()

    def split_hdf5(self):
        file = self.task_dir / "splits.hdf5"
        if self.notes:
            if self.discrete:
                k = "with_discrete_notes"
            else:
                k = "with_notes"
        else:
            k = "without_notes"
        if not Path.is_file(file):
            self.split(file, k)
        else:
            try:
                with h5py.File(file, "r") as hf:
                    hf[k]
            except KeyError:
                self.split(file, k)

    def prepare_data(self):
        self.xs_hdf5()
        if self.notes:
            self.note_hdf5()
        self.split_hdf5()

    def setup(self, stage: Optional[str] = None):
        hf = h5py.File(self.task_dir / "splits.hdf5", "r")
        if self.notes:
            if self.discrete:
                group = hf["with_discrete_notes"]
            else:
                group = hf["with_notes"]
        else:
            group = hf["without_notes"]
        if stage == "fit" or stage is None:
            self.train = EHRDataset(
                group["train"], self.notes, self.discrete, self.merge
            )
            # self.val = EHRDataset(group["val"], self.notes, self.discrete, self.merge)
            self.meta = EHRDataset(group["val"], self.notes, self.discrete, self.merge)
        if stage == "test" or stage is None:
            self.test = EHRDataset(group["test"], self.notes, self.discrete, self.merge)
        hf.close()

    def train_dataloader(self):
        # # target = self.train.y
        # # class_sample_count = np.array(
        # #     [len(np.where(target == t)[0]) for t in np.unique(target)]
        # # )
        # # weight = 1.0 / class_sample_count
        # # samples_weight = np.array([weight[t] for t in target])
        # # samples_weight = torch.from_numpy(samples_weight)
        # # samples_weight = samples_weight.double()
        # # sampler = WeightedRandomSampler(
        # #     weights=samples_weight, num_samples=1400, replacement=True
        # # )
        # minority_indices = [i for i, (x, y, idx) in enumerate(self.train) if y == 1]
        # minority_dataset = LabeledSubset(self.train, minority_indices)
        # # minority_dataset = [self.train[i] for i in minority_indices]
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def meta_dataloader(self):
        # create balanced validation set
        idx = np.where(np.array(self.meta.y) == 0)[0]
        np.random.shuffle(idx)
        idx = idx[10:]

        idx_1 = np.where(np.array(self.meta.y) == 1)[0]
        np.random.shuffle(idx_1)
        idx_1 = idx_1[10:]
        idx = np.concatenate((idx, idx_1))
        self.meta.X = np.delete(self.meta.X, idx, axis=0)
        self.meta.y = np.delete(self.meta.y, idx, axis=0)

        print(len(self.meta.y), len(self.meta.X))

        return DataLoader(
            self.meta,
            batch_size=20,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
