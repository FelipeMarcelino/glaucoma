import torch
import pandas as pd
import random
import os
from torchvision.io import read_image
from typing import List
from skimage import io
from params import ROOT_DIR
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, GroupKFold


class GlaucomaRandomDataset(Dataset):
    """Dataset contaning eyes fundus image"""

    def __init__(
        self,
        glaucoma_data: pd.DataFrame,
        root_dir: Path,
        ft_columns: List[str],
        double_img: bool,
        device: torch.device,
    ):
        super(GlaucomaRandomDataset, self).__init__()
        self.glaucoma_data = glaucoma_data
        self.root_dir = root_dir
        self.ft_columns = ft_columns
        self.double_img = double_img
        self.device = device

    def __len__(self):
        return len(self.glaucoma_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.glaucoma_data.iloc[idx]

        photo_1 = read_image(str(self.root_dir / row["photo_1"]))
        photo_2 = read_image(str(self.root_dir / row["photo_2"]))
        label = row["label"]
        eye_side = row["eye_side"]

        ft_numerical = row[self.ft_columns].to_numpy(dtype="float32")

        ft_numerical = torch.from_numpy(ft_numerical)

        if self.double_img:
            return photo_1, photo_2, ft_numerical, label
        else:
            # if random.random() > 0.5:
            #     return photo_1, photo_2, ft_numerical, label
            # else:
            #     return photo_2, photo_1, ft_numerical, label

            if random.random() > 0.5:
                return photo_1, photo_2, ft_numerical, label
            else:
                return photo_1, photo_2, ft_numerical, label


def init_dataloader(
    data,
    batch_size,
    ft_columns,
    double_img,
    device,
):

    dataset = GlaucomaRandomDataset(
        data,
        ROOT_DIR,
        ft_columns,
        double_img,
        device,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=int(os.getenv("NUM_WORKERS")),
    )

    return dataloader


def init_k_fold(data, n_splits):
    # kf = KFold(n_splits=n_splits, shuffle=True)
    group_kfold = GroupKFold(n_splits=n_splits)
    patients = data["Patient"].values.tolist()

    for train_index, test_index in group_kfold.split(data, groups=patients):
        train = data.iloc[train_index]
        test = data.iloc[test_index]

        yield train, test
