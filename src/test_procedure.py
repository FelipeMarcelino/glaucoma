import torch
import time
import pandas as pd
from params import ROOT_DIR
from sklearn.preprocessing import MinMaxScaler
from dataset import init_dataloader
from model import init_transforms
from test import get_sigmoid_pred, get_shap_values


def inference(
    path,
    model,
    numerical_columns,
    randaugop,
    input_size,
    data,
    oos_dataset_path,
    double_img_bool,
    output_tab,
    score,
    shap,
    size_shap,
    balanced_shap,
    model_id,
    model_folder,
    device,
    batch_size,
    model_name,
    train_loader_name,
    val_loader_name,
):
    print(model.load_state_dict(torch.load(path + model_name + ".pth")))

    train_loader = torch.load(path + train_loader_name + ".pth")
    val_loader = torch.load(path + val_loader_name + ".pth")
    val_loader.dataset.root_dir = ROOT_DIR
    train_loader.dataset.root_dir = ROOT_DIR

    train_dataset_transformed = train_loader.dataset.glaucoma_data

    oos_loader = None

    if oos_dataset_path:

        oos = pd.read_csv(oos_dataset_path)

        train = data.iloc[train_dataset_transformed.index]

        min_max_scaler = MinMaxScaler()

        train[numerical_columns] = min_max_scaler.fit_transform(
            train[numerical_columns]
        )

        oos[numerical_columns] = min_max_scaler.transform(oos[numerical_columns])

        (
            _,
            preprocessing_oos,
            preprocessing_tab,
        ) = init_transforms(input_size, randaugop)

        oos_loader = init_dataloader(
            oos,
            preprocessing_oos,
            preprocessing_tab,
            batch_size,
            numerical_columns,
            double_img_bool,
            device,
        )

    inference_ms = None
    if score:
        inference_time_start = time.time()
        pred_size = get_sigmoid_pred(
            model,
            train_loader,
            val_loader,
            oos_loader,
            output_tab,
            double_img_bool,
            model_id,
            model_folder,
            model_name,
        )
        inference_time_stop = time.time()
        inference_ms = round(
            (inference_time_stop - inference_time_start) / (pred_size * 3), 3
        )

    if shap:
        get_shap_values(
            model,
            train_loader,
            val_loader,
            oos_loader,
            balanced_shap,
            size_shap,
            input_size,
            path,
            model_name,
            double_img_bool,
            output_tab,
        )

    return inference_ms
