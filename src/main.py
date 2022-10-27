#!/usr/bin/env python
import click
import sys
import pickle
import torch
import random
import pandas as pd
import numpy as np
import uuid
import socket
import os


from sklearn.preprocessing import MinMaxScaler
from dataset import  init_k_fold
from model import init_model, init_transforms
from train import pre_train, train_model


# Fix seed's for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


@click.command
@click.argument(
    "csv_file", default="../data.csv", nargs=1, type=click.Path(exists=True)
)
@click.option(
    "--epochs",
    default=100,
    type=int,
    help="Total of epochs to train the model",
    show_default=True,
)
@click.option(
    "--test",
    default=False,
    type=bool,
    is_flag=True,
    show_default=True,
    help="Load model\
              and test it",
)
@click.option(
    "--backbone",
    default="regnet",
    type=click.Choice(
        ["regnetx", "regnet16x", "regnet32x",
         "mobile", "shuffle", "efficient",
         "vit", "inception", "resnet",
         "regnet", "regnet16", "regnet32"]
    ),
)
@click.option("--scratch", default=False, is_flag=True, type=bool)
@click.option("--feature_extract", default=False, type=bool, is_flag=True)
@click.option("--frac_val", default=0.2, type=float)
@click.option("--k_fold", default=-1, type=int)
@click.option("--debug", default=2, type=int)
@click.option(
    "--model_folder",
    default="../models/",
    type=str,
    help="Path to save model",
)
@click.option("--double_img", is_flag=True, default=False, type=bool)
@click.option("--output_tab", default=None, type=int)
@click.option("--early_start", default=50, type=int)
@click.option(
    "--optim",
    default="adam",
    type=click.Choice(["adam", "sgd", "radam", "ranger"]),
)
@click.option("--lr", default=0.0001, type=float)
@click.option("--batch_size", default=16, type=int)
@click.option("--patient", default=10, type=int)
@click.option("--overwrite", is_flag=True, default=False, type=bool)
def main(
    csv_file,
    epochs: int,
    test: bool,
    backbone: str,
    scratch: bool,
    feature_extract: bool,
    frac_val: float,
    k_fold: int,
    debug: int,
    model_folder: str,
    double_img: bool,
    output_tab: int,
    early_start: int,
    optim: str,
    lr: float,
    batch_size: int,
    patient: int,
    overwrite: bool,
):

    if backbone == "inception":
        is_inception = True
    else:
        is_inception = False

    params = {
        "epochs": epochs,
        #"scratch": scratch,
        "feature_extract": feature_extract,
        "frac_val": frac_val,
        "k_fold": k_fold if k_fold > 2 else None,
        "double_img": double_img,
        "output_tab": output_tab if output_tab else None,
        "early_start": early_start,
        "optim": optim,
        "lr": lr,
        #"batch_size": batch_size,
        "is_inception": is_inception,
        "backbone": backbone,
    }

    if not overwrite:
        try:
            temp_summary = pd.read_csv("../model_summary.csv", sep=",")
            query = ' and '.join([f'{k} == {repr(v)}' for k, v in params.items() if v is not None])
            query_rows = temp_summary.query(query)
            if len(query_rows) != 0:
                print("Model already tested!!! Exiting...")
                return
        except FileNotFoundError:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_id = uuid.uuid4().hex

    path = model_folder + str(model_id) + "/"

    os.makedirs(path)

    if scratch:
        pretrained = False
    else:
        pretrained = True

    # Loading data
    data = pd.read_csv(csv_file)

    numerical_columns = data.select_dtypes(include=np.number).columns.tolist()
    numerical_columns.remove("label")
    ft_size = len(numerical_columns)

    # Saving results into list
    fold_val_loss_history = []
    fold_val_acc_history = []
    fold_val_auc_history = []
    fold_val_sensitivity_history = []
    fold_val_specificity_history = []
    fold_train_loss_history = []
    fold_train_acc_history = []
    fold_train_auc_history = []
    fold_train_sensitivity_history = []
    fold_train_specificity_history = []

    min_max_scaler = MinMaxScaler()

    if k_fold >= 2:
        # FIXME: Separar por paciente e não por olho
        folds = init_k_fold(data, k_fold)

        for index, (train, val) in enumerate(folds):
            print("Fold:", index + 1)

            train[numerical_columns] = min_max_scaler.fit_transform(
                train[numerical_columns]
            )

            val[numerical_columns] = min_max_scaler.transform(val[numerical_columns])

            model, input_size = init_model(
                backbone,
                pretrained,
                feature_extract,
                double_img,
                output_tab,
                ft_size,
            )
            preprocessing_train, preprocessing_val, preprocessing_tab = init_transforms(
                input_size
            )

            model, optimizer, criterion, dataloader_train, dataloader_val = pre_train(
                train,
                val,
                preprocessing_train,
                preprocessing_val,
                preprocessing_tab,
                batch_size,
                model,
                feature_extract,
                device,
                debug,
                numerical_columns,
                double_img,
                optim,
                lr,
            )
            dataloaders_dict = {}
            dataloaders_dict["train"] = dataloader_train
            dataloaders_dict["val"] = dataloader_val
            (
                model,
                val_loss_history,
                val_acc_history,
                val_auc_history,
                val_sensitivity_history,
                val_specificity_history,
                train_loss_history,
                train_acc_history,
                train_auc_history,
                train_sensitivity_history,
                train_specificity_history,
            ) = train_model(
                model,
                dataloaders_dict,
                criterion,
                optimizer,
                device,
                double_img,
                output_tab,
                early_start,
                epochs,
                is_inception=is_inception,
                patient=patient
            )

            torch.save(model.state_dict(), path + str(model_id) + ".pth")

            torch.save(
                dataloader_train,
                path + "train_dataloader_" + str(model_id) + ".pth",
            )
            torch.save(
                dataloader_val,
                path + "val_dataloader_" + str(model_id) + ".pth",
            )

            fold_val_loss_history.append(val_loss_history)
            fold_val_acc_history.append(val_acc_history)
            fold_val_auc_history.append(val_auc_history)
            fold_val_sensitivity_history.append(val_sensitivity_history)
            fold_val_specificity_history.append(val_specificity_history)
            fold_train_loss_history.append(train_loss_history)
            fold_train_acc_history.append(train_acc_history)
            fold_train_auc_history.append(train_auc_history)
            fold_train_sensitivity_history.append(train_sensitivity_history)
            fold_train_specificity_history.append(train_specificity_history)
    else:
        model, input_size = init_model(
            backbone,
            pretrained,
            feature_extract,
            double_img,
            output_tab,
            ft_size,
        )
        preprocessing_train, preprocessing_val, preprocessing_tab = init_transforms(
            input_size
        )

        # FIXME: Separar por paciente e não por olho
        msk = np.random.rand(len(data)) < (1 - frac_val)
        train = data[msk]
        val = data[~msk]

        train[numerical_columns] = min_max_scaler.fit_transform(
            train[numerical_columns]
        )
        val[numerical_columns] = min_max_scaler.transform(val[numerical_columns])

        model, optimizer, criterion, dataloader_train, dataloader_val = pre_train(
            train,
            val,
            preprocessing_train,
            preprocessing_val,
            preprocessing_tab,
            batch_size,
            model,
            feature_extract,
            device,
            debug,
            numerical_columns,
            double_img,
            optim,
            lr,
        )

        torch.save(
            dataloader_train,
            path + "train_dataloader_" + str(model_id) + ".pth",
        )
        torch.save(
            dataloader_val,
            path + "val_dataloader_" + str(model_id) + ".pth",
        )

        dataloaders_dict = {}
        dataloaders_dict["train"] = dataloader_train
        dataloaders_dict["val"] = dataloader_val

        try:
            (
                model,
                val_loss_history,
                val_acc_history,
                val_auc_history,
                val_sensitivity_history,
                val_specificity_history,
                train_loss_history,
                train_acc_history,
                train_auc_history,
                train_sensitivity_history,
                train_specificity_history,
            ) = train_model(
                model,
                dataloaders_dict,
                criterion,
                optimizer,
                device,
                double_img,
                output_tab,
                early_start,
                epochs,
                is_inception=is_inception,
                patient=patient,
            )
        except KeyboardInterrupt:
            pass

        fold_val_loss_history.append(val_loss_history)
        fold_val_acc_history.append(val_acc_history)
        fold_val_auc_history.append(val_auc_history)
        fold_val_sensitivity_history.append(val_sensitivity_history)
        fold_val_specificity_history.append(val_specificity_history)
        fold_train_loss_history.append(train_loss_history)
        fold_train_acc_history.append(train_acc_history)
        fold_train_auc_history.append(train_auc_history)
        fold_train_sensitivity_history.append(train_sensitivity_history)
        fold_train_specificity_history.append(train_specificity_history)

        torch.save(model.state_dict(), path + str(model_id) + ".pth")

    dict_results = {}
    dict_results["val_loss_history"] = fold_val_loss_history
    dict_results["val_acc_history"] = fold_val_acc_history
    dict_results["val_auc_history"] = fold_val_auc_history
    dict_results["val_sensitivity_history"] = fold_val_sensitivity_history
    dict_results["val_specificity_history"] = fold_val_specificity_history
    dict_results["train_loss_history"] = fold_train_loss_history
    dict_results["train_acc_history"] = fold_train_acc_history
    dict_results["train_auc_history"] = fold_train_auc_history
    dict_results["train_sensitivity_history"] = fold_train_sensitivity_history
    dict_results["train_specificity_history"] = fold_train_specificity_history

    with open(path + str(model_id), "wb") as handle:
        pickle.dump(dict_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path + str(model_id), "wb") as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    row_data = {
        "model_id": model_id,
        "k_fold": k_fold if k_fold > 2 else np.nan,
        "frac_val": frac_val if k_fold < 2 else np.nan,
        "val_best_acc": np.max(dict_results["val_acc_history"]),
        "val_best_auc": np.max(dict_results["val_auc_history"]),
        "val_best_sp": np.max(dict_results["val_specificity_history"]),
        "val_best_sn": np.max(dict_results["val_sensitivity_history"]),
        "val_avg_acc": np.mean(dict_results["val_acc_history"]),
        "val_avg_auc": np.mean(dict_results["val_auc_history"]),
        "val_avg_sp": np.mean(dict_results["val_specificity_history"]),
        "val_avg_sn": np.mean(dict_results["val_sensitivity_history"]),
        "train_best_acc": np.max(dict_results["train_acc_history"]),
        "train_best_auc": np.max(dict_results["train_auc_history"]),
        "train_best_sp": np.max(dict_results["train_specificity_history"]),
        "train_best_sn": np.max(dict_results["train_sensitivity_history"]),
        "train_avg_acc": np.mean(dict_results["train_acc_history"]),
        "train_avg_auc": np.mean(dict_results["train_auc_history"]),
        "train_avg_sp": np.mean(dict_results["train_specificity_history"]),
        "train_avg_sn": np.mean(dict_results["train_sensitivity_history"]),
        "host_name": socket.gethostname(),
        "is_inception": is_inception,
        "optim": optim,
        "lr": lr,
        "epochs": epochs,
        "double_img": 1 if double_img is True else 0,
        "output_tab": output_tab if output_tab is not None else np.nan,
        "backbone": backbone,
        "feature_extract": feature_extract,
        "early_start": early_start,
    }

    try:
        summary = pd.read_csv("../model_summary.csv", sep=",")
    except FileNotFoundError:
        row = pd.DataFrame(row_data, index=[0])
        summary = row
    else:
        row = pd.DataFrame(row_data, index=[0])
        summary = pd.concat([summary, row])

    summary.to_csv("../model_summary.csv", index=False)


if __name__ == "__main__":
    main()
