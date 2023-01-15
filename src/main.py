# TODO: adding time cross validation and total, datetime.now, and torch/torchvision version. Add
# shallow copy to double img shared parameters model.
#!/usr/bin/env python
import click
import pickle
import subprocess
from pytorch_memlab import LineProfiler
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
from dataset import init_k_fold
from model import init_model, init_transforms
from train import pre_train, train_model
from params import SUMMARY_PATH, DF_PEAK_SUMMARY
from utils import calculate_mem_size, check_execution_already
from dotenv import load_dotenv
from test_procedure import inference

load_dotenv()


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
    default="regnety",
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
            "regnety",
            "regnet16y",
            "regnet32y",
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
@click.option("-sc", "--scheduler_name", default=None, type=click.Choice(["plateau"]))
@click.option("--lr", default=0.0001, type=float)
@click.option("--batch_size", default=16, type=int)
@click.option("--patient_el", default=10, type=int)
@click.option("--overwrite", is_flag=True, default=False, type=bool)
@click.option("--autocast", is_flag=True, default=False, type=bool)
@click.option("--cudnn_bench", is_flag=True, default=False, type=bool)
@click.option("-aug", "--randaugop", type=int, default=0)
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
    scheduler_name: str,
    lr: float,
    batch_size: int,
    patient_el: int,
    overwrite: bool,
    autocast: bool,
    cudnn_bench: bool,
    randaugop: int,
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
        "scheduler": scheduler_name,
        "randaugop": randaugop,
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
        fold_lr_history = []

        min_max_scaler = MinMaxScaler()

        total_cross_val_time = 0

        if k_fold >= 2:
            total_cross_val_time = 0
            folds = init_k_fold(data, k_fold)

            for index, (train, val) in enumerate(folds):
                start_fold = time.time()
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
                ) = init_transforms(input_size, randaugop)

                if debug:
                    print(torchsummary(model))

                (
                    model,
                    optimizer,
                    scheduler,
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
                    scheduler_name,
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
                    lr_history,
                ) = train_model(
                    model,
                    dataloaders_dict,
                    criterion,
                    optimizer,
                    scheduler,
                    device,
                    double_img,
                    output_tab,
                    is_inception,
                    early_start,
                    epochs,
                    patient_el=patient_el,
                    autocast=autocast,
                    cudnn_bench=cudnn_bench,
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
                fold_lr_history.append(lr_history)

                stop_fold = time.time()
                total_cross_val_time_iter = stop_fold - start_fold
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
                input_size,
                randaugop,
            )

            # msk = np.random.rand(len(data)) < (1 - frac_val)

            splitter = GroupShuffleSplit(
                test_size=frac_val, n_splits=1, random_state=42
            )
            split = splitter.split(data, groups=data["Patient"])
            train_inds, test_inds = next(split)

            train = data.iloc[train_inds]
            val = data.iloc[test_inds]

            # train = data[msk]
            # val = data[~msk]

            train[numerical_columns] = min_max_scaler.fit_transform(
                train[numerical_columns]
            )
            val[numerical_columns] = min_max_scaler.transform(val[numerical_columns])

            (
                model,
                optimizer,
                scheduler,
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
                scheduler_name,
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
                if debug:
                    debug_params = {}
                    debug_params["host_name"] = socket.gethostname()
                    debug_params["backbone"] = backbone
                    debug_params["batch_size"] = batch_size
                    debug_params["mixed_precision"] = 1 if autocast else 0
                    debug_params["cudnn_bench"] = 1 if cudnn_bench else 0
                    debug_params["double_img"] = double_img
                    debug_params["output_tab"] = output_tab
                    debug_params["torch_version"] = torch.__version__
                    debug_params["torchvision_version"] = torchvision.__version__
                    debug_params["cuda_version"] = float(torch.version.cuda)
                    debug_params["torchvision_cuda_version"] = int(
                        torchvision.version.cuda
                    )
                    debug_params["running_cuda"] = float(
                        (
                            subprocess.check_output(["nvidia-smi"])
                            .decode()
                            .split("\n")[2]
                            .split(" ")[-6]
                        )
                    )

                    if (
                        check_execution_already(debug_params, DF_PEAK_SUMMARY)
                        and not overwrite
                    ):
                        print("Model/Debug already tested!!! Exiting...")
                        return 0

                    with LineProfiler(train_model) as prof:
                        torch.cuda.empty_cache()
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
                            lr_history,
                        ) = train_model(
                            model,
                            dataloaders_dict,
                            criterion,
                            optimizer,
                            scheduler,
                            device,
                            double_img,
                            output_tab,
                            is_inception,
                            early_start,
                            epochs,
                            patient_el=patient_el,
                            autocast=autocast,
                            cudnn_bench=cudnn_bench,
                        )
                        df_peak = pd.read_html(prof.display()._repr_html_())[0]
                        df_peak.columns = df_peak.columns.droplevel([1, 2])
                        df_peak["host_name"] = socket.gethostname()
                        df_peak["backbone"] = backbone
                        df_peak["batch_size"] = batch_size
                        df_peak["mixed_precision"] = 1 if autocast else 0
                        df_peak["cudnn_bench"] = 1 if cudnn_bench else 0
                        df_peak["double_img"] = double_img
                        df_peak["output_tab"] = output_tab
                        df_peak["torch_version"] = torch.__version__
                        df_peak["torchvision_version"] = torchvision.__version__
                        df_peak["cuda_version"] = torch.version.cuda
                        df_peak["torchvision_cuda_version"] = torchvision.version.cuda
                        df_peak["running_cuda"] = (
                            subprocess.check_output(["nvidia-smi"])
                            .decode()
                            .split("\n")[2]
                            .split(" ")[-6]
                        )

                        df_peak["total_mem_mb"] = df_peak.apply(
                            lambda x: calculate_mem_size(
                                x["active_bytes"], x["reserved_bytes"]
                            ),
                            axis=1,
                        )
                        df_peak.drop(
                            columns=["line", "code", "active_bytes", "reserved_bytes"],
                            inplace=True,
                        )

                        df_peak_row = df_peak[
                            df_peak["total_mem_mb"] == df_peak["total_mem_mb"].max()
                        ]

                        df_peak_mem_summary = pd.read_csv(DF_PEAK_SUMMARY)

                        df_peak_mem_summary = pd.concat(
                            [df_peak_mem_summary, df_peak_row]
                        )

                        df_peak_mem_summary.to_csv(DF_PEAK_SUMMARY, index=False)
                        return 0

                else:
                    torch.cuda.empty_cache()
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
                        lr_history,
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
                        patient_el=patient_el,
                        autocast=autocast,
                        cudnn_bench=cudnn_bench,
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
                fold_lr_history.append(lr_history)

                torch.save(model.state_dict(), path + "model" + ".pth")
            except KeyboardInterrupt:
                return 0

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
        dict_results["lr_history"] = fold_lr_history

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
            "patient_el": patient_el,
            "output_tab": output_tab if output_tab is not None else np.nan,
            "scheduler": scheduler_name if scheduler_name is not None else np.nan,
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
            "randaugop": randaugop,
            "inference_time": np.nan,
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
        summary: pd.DataFrame = summary.drop_duplicates(subset=["model_id"])
        print(f"Testing model...")
        print(model_id)
        print(summary.shape)
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
        inference_ms_all = None

        if score:
            if (
                row["pred_val"].values == 1
                and row["pred_oos"].values == 1
                and not overwrite
            ):
                score = False
            if row["pred_val"].values == 1 and not oos_dataset_path and not overwrite:
                score = False

        if shap:
            if (
                row["shap_val"].values == 1
                and row["shap_oos"].values == 1
                and not overwrite
            ):
                shap = False
            if row["shap_val"].values == 1 and not oos_dataset_path and not overwrite:
                shap = False

        if not shap and not score and not overwrite:
            print("Already calculated!!!")
            return 0

        if row["frac_val"].values[0] is not pd.NA:
            model_name = "model"
            train_loader_name = "train_dataloader"
            val_loader_name = "val_dataloader"

            model, input_size = init_model(
                backbone,
                pretrained,
                double_img_bool,
                output_tab,
                ft_size,
            )
            model.to(device)

            inference_ms = inference(
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
                ft_size,
            )
        else:
            k_fold = int(row["k_fold"])

            for i in range(1, k_fold + 1):

                model, input_size = init_model(
                    backbone,
                    pretrained,
                    double_img_bool,
                    output_tab,
                    ft_size,
                )
                model_name = "model_fold_" + str(i)
                train_loader_name = "train_dataloader_fold_" + str(i)
                val_loader_name = "val_dataloader_fold_" + str(i)
                model.to(device)

                inference_ms = inference(
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
                    ft_size,
                )
                if not inference_ms_all:
                    inference_ms_all = inference_ms

        if inference_ms:
            summary.loc[
                summary["model_id"] == model_id, "inference_time"
            ] = inference_ms_all
        if score:
            summary.loc[summary["model_id"] == model_id, "pred_val"] = 1

            if oos_dataset_path:
                summary.loc[summary["model_id"] == model_id, "pred_oos"] = 1
        if shap:
            summary.loc[summary["model_id"] == model_id, "shap_val"] = 1

            if oos_dataset_path:
                summary.loc[summary["model_id"] == model_id, "shap_oos"] = 1

        summary.to_csv("../model_summary.csv", index=False)


if __name__ == "__main__":
    main()
