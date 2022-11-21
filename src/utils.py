import sys
import numpy as np
import pandas as pd
import torch
import subprocess
import torchvision
from typing import List
from itertools import product
from pathlib import Path
from typing import Dict
from params import (
    BACKBONE_ARG,
    BATCHSIZE_ARG,
    K_FOLD_ARG,
    OPTIM_ARG,
    FRAC_VAL_ARG,
    DOUBLE_IMG_ARG,
    EARLY_START_ARG,
    OUTPUT_TAB_ARG,
    LR_ARG,
    EPOCHS_ARG,
)


def check_execution_already(params: Dict, summary_path: Path) -> bool:

    try:
        temp_summary = pd.read_csv(summary_path, sep=",")
        query = " and ".join(
            [
                f"{k} == {repr(v)}" if v is not np.nan else f"{k}.isnull()"
                for k, v in params.items()
            ]
        )
        query_rows = temp_summary.query(query)
        if len(query_rows) != 0:
            return True
        else:
            return False
    except FileNotFoundError:
        return False


def list_status_models_tested(
    params: Dict, double_img: bool, summary_path: Path
) -> pd.DataFrame:

    keys, values = zip(*params.items())
    permutation_params = [dict(zip(keys, v)) for v in product(*values)]

    for p_params in permutation_params:
        if double_img:
            p_params["double_img"] = 1.0
        else:
            p_params["double_img"] = 0.0
        p_params["torch_version"] = torch.__version__
        p_params["torchvision_version"] = torchvision.__version__
        executed = check_execution_already(p_params, summary_path)
        p_params["executed"] = 1 if executed else 0

    permutation_status = pd.DataFrame(permutation_params)

    return permutation_status


def create_commands(filtered_status: pd.DataFrame):

    commands = []
    for _, row in filtered_status.iterrows():
        command_base = ""
        command_base += BACKBONE_ARG + " " + row["backbone"] + " "
        command_base += LR_ARG + " " + str(row["lr"]) + " "
        command_base += OPTIM_ARG + " " + row["optim"] + " "
        command_base += BATCHSIZE_ARG + " " + str(row["batch_size"]) + " "
        command_base += EPOCHS_ARG + " " + str(row["epochs"]) + " "
        command_base += EARLY_START_ARG + " " + str(row["early_start"]) + " "
        command_base += OUTPUT_TAB_ARG + " " + str(row["output_tab"]) + " "
        if int(row["double_img"]) == 1:
            command_base += DOUBLE_IMG_ARG + " "
        if "frac_val" in row:
            command_base += FRAC_VAL_ARG + " " + str(row["frac_val"])
        if "k_fold" in row:
            command_base += K_FOLD_ARG + " " + str(row["k_fold"])

        commands.append(command_base)

    return commands


def check_already_running(commands: List[str]):

    pythonProcess = subprocess.check_output(
        "ps -ef  | grep felipema |  grep main.py",
        shell=True,
    ).decode()
    pythonProcess = pythonProcess.split("\n")

    command_selected = None
    for command in commands:
        exclude_command = False
        for process in pythonProcess:
            if command in process:
                exclude_command = True
                break
        if exclude_command:
            continue
        else:
            command_selected = command
            break

    return command_selected
