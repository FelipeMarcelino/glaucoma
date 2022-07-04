import torch
import pandas as pd
import random
from typing import List
from skimage import io
from params import ROOT_DIR
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold


class GlaucomaRandomDataset(Dataset):
    """Dataset contaning eyes fundus image"""

    def __init__(
        self,
        glaucoma_data: pd.DataFrame,
        root_dir: Path,
        ft_columns: List[str],
        double_img: bool,
        transform_img=None,
        transform_tab=None,
    ):
        super(GlaucomaRandomDataset, self).__init__()
        self.glaucoma_data = glaucoma_data
        self.root_dir = root_dir
        self.transform_img = transform_img
        self.transform_tab = transform_tab
        self.ft_columns = ft_columns
        self.double_img = double_img

    def __len__(self):
        return len(self.glaucoma_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.glaucoma_data.iloc[idx]

        photo_1 = io.imread(row["photo_1"])
        photo_2 = io.imread(row["photo_2"])
        label = row["label"]
        eye_side = row["eye_side"]

        ft_numerical = row[self.ft_columns].to_numpy(dtype="float32")

        # if self.transform_tab:
        #     ft_numerical = self.transform_tab(ft_numerical)
        ft_numerical = torch.from_numpy(ft_numerical)

        if self.transform_img:
            photo_1 = self.transform_img(photo_1)
            photo_2 = self.transform_img(photo_2)

        if self.double_img:
            return photo_1, photo_2, ft_numerical, label
        else:
            if random.random() > 0.5:
                return photo_1, photo_2, ft_numerical, label
            else:
                return photo_2, photo_1, ft_numerical, label


def init_dataloader(
    data, preprocessing_img, preprocessing_tab, batch_size, ft_columns, double_img
):

    dataset = GlaucomaRandomDataset(
        data, ROOT_DIR, ft_columns, double_img, preprocessing_img, preprocessing_tab
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def init_k_fold(data, n_splits):
    kf = KFold(n_splits=n_splits, shuffle=True)

    for train_index, test_index in kf.split(data):
        train = data.iloc[train_index]
        test = data.iloc[test_index]

        yield train, test
