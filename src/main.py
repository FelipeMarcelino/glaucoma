# TODO: adding time cross validation and total, datetime.now, and torch/torchvision version. Add
# shallow copy to double img shared parameters model.
#!/usr/bin/env python
import click
import pickle
import torch
import random
import pandas as pd
import numpy as np
import uuid
import socket
import os
import time
import sys


from pathlib import Path
from datetime import datetime
from torchinfo import summary as torchsummary
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit
from torchvision.ops.boxes import torchvision
from dataset import init_dataloader, init_k_fold
from model import init_model, init_transforms
from test import get_sigmoid_pred, get_shap_values
from train import pre_train, train_model
from params import ROOT_DIR, SUMMARY_PATH
from utils import check_execution_already


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
    "--score",
    default=False,
    type=bool,
    is_flag=True,
    show_default=True,
    help="Load model\
              and score data",
)
@click.option(
    "--shap",
    default=False,
    type=bool,
    is_flag=True,
    show_default=True,
    help="Load model and calculate shap",
)
@click.option(
    "--balanced_shap",
    default=False,
    type=bool,
    is_flag=True,
    show_default=True,
    help="Calculate shap using a balanced dataset",
)
@click.option(
    "--size_shap",
    default=50,
    type=int,
    help="The size of shap background dataset",
)
@click.option("--oos_dataset_path", "-oos", type=click.Path(exists=True), default=None)
@click.option(
    "--backbone",
    default="regnet",
    type=click.Choice(
        [
            "regnetx",
            "regnet16x",
            "regnet32x",
            "mobile",
            "shuffle",
            "efficient",
            "vit",
            "inception",
            "resnet",
            "regnet",
            "regnet16",
            "regnet32",
        ]
    ),
)
@click.option("--frac_val", default=0.2, type=float)
@click.option("--k_fold", default=-1, type=int)
@click.option("--debug", is_flag=True, default=False, type=bool)
@click.option("--model_id", default=None, type=str)
@click.option(
    "--model_folder",
    default="../models/",
    type=str,
    help="Path to save model",
)
@click.option("--double_img", is_flag=True, default=False, type=bool)
@click.option("--output_tab", default=0, type=int)
@click.option("--early_start", default=100, type=int)
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
    score: bool,
    shap: bool,
    balanced_shap: bool,
    size_shap: int,
    oos_dataset_path: Path,
    backbone: str,
    frac_val: float,
    k_fold: int,
    debug: bool,
    model_id: str,
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

    start = time.time()
    test = False

    if score or shap:
        test = True

    params = {
        "epochs": epochs,
        "frac_val": frac_val if k_fold < 2 else np.nan,
        "k_fold": k_fold if k_fold > 2 else np.nan,
        "double_img": 1.0 if double_img is True else 0.0,
        "output_tab": output_tab,
        "early_start": early_start,
        "optim": optim,
        "lr": lr,
        "batch_size": batch_size,
        "backbone": backbone,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }

    if not test and not debug and not overwrite:
        if check_execution_already(params, SUMMARY_PATH):
            print("Model already tested!!! Exiting...")
            return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not test:
        model_id = uuid.uuid4().hex

        path = model_folder + str(model_id) + "/"

        os.makedirs(path)

        pretrained = True

    # Loading data
    data = pd.read_csv(csv_file)

    numerical_columns = data.select_dtypes(include=np.number).columns.tolist()
    numerical_columns.remove("label")
    ft_size = len(numerical_columns)

    if not test:
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

        total_cross_val_time = 0

        if k_fold >= 2:
            total_cross_val_time = 0
            # FIXME: Separar por paciente e não por olho
            folds = init_k_fold(data, k_fold)

            for index, (train, val) in enumerate(folds):
                start = time.time()
                print("Fold:", index + 1)

                train[numerical_columns] = min_max_scaler.fit_transform(
                    train[numerical_columns]
                )

                val[numerical_columns] = min_max_scaler.transform(
                    val[numerical_columns]
                )

                model, input_size = init_model(
                    backbone,
                    pretrained,
                    double_img,
                    output_tab,
                    ft_size,
                )
                (
                    preprocessing_train,
                    preprocessing_val,
                    preprocessing_tab,
                ) = init_transforms(input_size)

                if debug:
                    print(torchsummary(model))

                (
                    model,
                    optimizer,
                    criterion,
                    dataloader_train,
                    dataloader_val,
                ) = pre_train(
                    train,
                    val,
                    preprocessing_train,
                    preprocessing_val,
                    preprocessing_tab,
                    batch_size,
                    model,
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

                if backbone == "inception":
                    is_inception = True
                else:
                    is_inception = False
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
                    is_inception,
                    early_start,
                    epochs,
                    patient=patient,
                )

                torch.save(
                    model.state_dict(), path + "model_fold_" + str(index + 1) + ".pth"
                )

                torch.save(
                    dataloader_train,
                    path + "train_dataloader_fold_" + str(index + 1) + ".pth",
                )
                torch.save(
                    dataloader_val,
                    path + "val_dataloader_fold_" + str(index + 1) + ".pth",
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

                stop = time.time()
                total_cross_val_time_iter = stop - start
                total_cross_val_time += total_cross_val_time_iter
        else:
            model, input_size = init_model(
                backbone,
                pretrained,
                double_img,
                output_tab,
                ft_size,
            )

            if debug:
                print(torchsummary(model))
                print(model)

            preprocessing_train, preprocessing_val, preprocessing_tab = init_transforms(
                input_size
            )

            # FIXME: Separar por paciente e não por olho
            msk = np.random.rand(len(data)) < (1 - frac_val)

            # FIXME: Remove comments
            # splitter = GroupShuffleSplit(test_size=frac_val, n_splits=1, random_state=42)
            # split = splitter.split(data, groups=data["Patient"])
            # train_inds, test_inds = next(split)

            # train = data.iloc[train_inds]
            # val = data[test_inds]
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
                device,
                debug,
                numerical_columns,
                double_img,
                optim,
                lr,
            )

            torch.save(
                dataloader_train,
                path + "train_dataloader.pth",
            )
            torch.save(
                dataloader_val,
                path + "val_dataloader.pth",
            )

            dataloaders_dict = {}
            dataloaders_dict["train"] = dataloader_train
            dataloaders_dict["val"] = dataloader_val

            if backbone == "inception":
                is_inception = True
            else:
                is_inception = False

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
                    is_inception,
                    early_start,
                    epochs,
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

            torch.save(model.state_dict(), path + "model" + ".pth")

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

        with open(path + "results.pkl", "wb") as handle:
            pickle.dump(dict_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        stop = time.time()

        total_time = stop - start
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)

        total_time_str = "{:0>2}:{:0>2}:{:05.2f}".format(
            int(hours), int(minutes), seconds
        )

        if k_fold > 2:
            hours_cross, rem_cross = divmod(total_cross_val_time / k_fold, 3600)
            minutes_cross, seconds_cross = divmod(rem_cross, 60)

            total_time_str_cross = "{:0>2}:{:0>2}:{:05.2f}".format(
                int(hours_cross), int(minutes_cross), seconds_cross
            )
        else:
            total_time_str_cross = total_time_str

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
            "optim": optim,
            "lr": lr,
            "epochs": epochs,
            "double_img": 1 if double_img is True else 0,
            "output_tab": output_tab if output_tab is not None else np.nan,
            "backbone": backbone,
            "early_start": early_start,
            "timestamp": str(datetime.now()),
            "total_hours": total_time_str,
            "average_cross_hours": total_time_str_cross,
            "torchvision_version": torchvision.__version__,
            "torch_version": torch.__version__,
            "history_added": 0,
            "shap_val": 0,
            "shap_oos": 0,
            "pred_val": 0,
            "pred_oos": 0,
            "batch_size": batch_size,
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
    else:
        if not model_id:
            print("Give a specific model id to test a new model. Exiting...")
            return

        summary = pd.read_csv("../model_summary.csv", sep=",")
        summary = summary.drop_duplicates(subset=["model_id"])
        print(f"Testing model...")
        row = summary[summary["model_id"] == model_id]
        print(row.squeeze())

        backbone = row["backbone"].values[0]
        pretrained = False

        path = model_folder + str(model_id) + "/"

        output_tab = (
            int(row["output_tab"].values[0])
            if row["output_tab"].values[0] > 0
            else None
        )
        double_img_bool = True if row["double_img"].values[0] > 0 else False
        model, input_size = init_model(
            backbone,
            pretrained,
            double_img_bool,
            output_tab,
            ft_size,
        )

        print(model.load_state_dict(torch.load(path + "model" + ".pth")))

        train_loader = torch.load(path + "train_dataloader" + ".pth")
        val_loader = torch.load(path + "val_dataloader" + ".pth")
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
                preprocessing_train,
                preprocessing_oos,
                preprocessing_tab,
            ) = init_transforms(input_size)

            oos_loader = init_dataloader(
                oos,
                preprocessing_oos,
                preprocessing_tab,
                batch_size,
                numerical_columns,
                double_img_bool,
                device,
            )

        if score:
            get_sigmoid_pred(
                model,
                train_loader,
                val_loader,
                oos_loader,
                output_tab,
                double_img_bool,
                model_id,
                model_folder,
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
            )


if __name__ == "__main__":
    main()
