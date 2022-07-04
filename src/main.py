import click
import pickle
import sys
import torch
import random
import torch.nn as nn
import pandas as pd
import numpy as np


from sklearn.preprocessing import MinMaxScaler
from dataset import init_dataloader, init_k_fold
from model import init_model, init_optimizer, init_transforms
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
    default=10,
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
    "--model_name",
    default="regnet",
    type=click.Choice(["regnet", "mobile", "shuffle", "efficient"]),
)
@click.option("--scratch", default=False, is_flag=True, type=bool)
@click.option("--feature_extract", default=False, type=bool, is_flag=True)
@click.option("--frac_val", default=0.2, type=float)
@click.option("--k_fold", default=-1, type=int)
@click.option("--debug", default=2, type=int)
@click.option(
    "--path",
    default="../models/",
    type=str,
    help="Path to save model",
)
@click.option("--multi_input", is_flag=True, default=False, type=bool)
@click.option("--double_img", is_flag=True, default=False, type=bool)
@click.option("--output_tab", default=3, type=int)
def main(
    csv_file,
    epochs: int,
    test: bool,
    model_name: str,
    scratch: bool,
    feature_extract: bool,
    frac_val: float,
    k_fold: int,
    debug: int,
    path: str,
    multi_input: bool,
    double_img: bool,
    output_tab: int,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 16

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
    fold_val_acc_history = []
    fold_val_auc_history = []
    fold_val_sensitivity_history = []
    fold_val_specificity_history = []

    min_max_scaler = MinMaxScaler()
    bk_model_name = model_name

    if k_fold >= 2:
        folds = init_k_fold(data, k_fold)

        for index, (train, val) in enumerate(folds):
            print("Fold:", index + 1)

            model_name = bk_model_name

            train[numerical_columns] = min_max_scaler.fit_transform(
                train[numerical_columns]
            )

            val[numerical_columns] = min_max_scaler.transform(val[numerical_columns])

            model, input_size = init_model(
                model_name,
                pretrained,
                feature_extract,
                multi_input,
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
            )
            dataloaders_dict = {}
            dataloaders_dict["train"] = dataloader_train
            dataloaders_dict["val"] = dataloader_val
            (
                model,
                val_acc_history,
                val_auc_history,
                val_sensitivity_history,
                val_specificity_history,
            ) = train_model(
                model,
                dataloaders_dict,
                criterion,
                optimizer,
                device,
                epochs,
                multi_input=multi_input,
            )

            if multi_input:
                model_name += "_multi_input"
            else:
                model_name += "_single_intput"

            if double_img:
                model_name += "_double_img"

            model_name = model_name + "_" + str(index + 1) + "k_fold"

            torch.save(model.state_dict(), path + model_name + ".pth")

            torch.save(
                dataloader_train,
                "../models/train_dataloader_" + model_name + ".pth",
            )
            torch.save(
                dataloader_val,
                "../models/val_dataloader_" + model_name + ".pth",
            )

            fold_val_acc_history.append(val_acc_history)
            fold_val_auc_history.append(val_auc_history)
            fold_val_sensitivity_history.append(val_sensitivity_history)
            fold_val_specificity_history.append(val_specificity_history)
    else:
        model, input_size = init_model(
            model_name,
            pretrained,
            feature_extract,
            multi_input,
            double_img,
            output_tab,
            ft_size,
        )
        preprocessing_train, preprocessing_val, preprocessing_tab = init_transforms(
            input_size
        )
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
        )

        if multi_input:
            model_name += "_multi_input_non_fold"
        else:
            model_name += "_single_intput_non_fold"

        if double_img:
            model_name += "_double_img"

        torch.save(
            dataloader_train,
            "../models/train_dataloader_" + model_name + ".pth",
        )
        torch.save(
            dataloader_val,
            "../models/val_dataloader_" + model_name + ".pth",
        )

        dataloaders_dict = {}
        dataloaders_dict["train"] = dataloader_train
        dataloaders_dict["val"] = dataloader_val
        (
            model,
            val_acc_history,
            val_auc_history,
            val_sensitivity_history,
            val_specificity_history,
        ) = train_model(
            model,
            dataloaders_dict,
            criterion,
            optimizer,
            device,
            epochs,
            multi_input=multi_input,
        )
        fold_val_acc_history.append(val_acc_history)
        fold_val_auc_history.append(val_auc_history)
        fold_val_sensitivity_history.append(val_sensitivity_history)
        fold_val_specificity_history.append(val_specificity_history)

        torch.save(model.state_dict(), path + model_name + ".pth")

    dict_results = {}
    dict_results["val_acc_history"] = fold_val_acc_history
    dict_results["val_auc_history"] = fold_val_auc_history
    dict_results["val_sensitivity_history"] = fold_val_sensitivity_history
    dict_results["val_specificity_history"] = fold_val_specificity_history

    if multi_input:
        bk_model_name += "_multi_input"
    else:
        bk_model_name += "_single_intput"

    if double_img:
        bk_model_name += "_double_img"

    if k_fold > 1:
        bk_model_name += "_k_fold"
    else:
        bk_model_name += "_non_k_fold"

    result_name = bk_model_name + ".pkl"
    with open(path + result_name, "wb") as handle:
        pickle.dump(dict_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
