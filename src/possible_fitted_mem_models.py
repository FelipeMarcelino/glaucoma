import pandas as pd
from typing import List
import click
import sys
from pandas import DataFrame
from params import SUMMARY_PATH, MODEL_USAGE_MEM

from utils import create_commands, list_status_models_tested, check_already_running

SAFE_MEM_REDUCE = 100

@click.command
@click.option(
    "--backbone",
    "-bb",
    default=["regnet"],
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
    multiple=True,
)
@click.option(
    "--optim",
    "-op",
    default=["adam"],
    type=click.Choice(["adam", "sgd", "radam", "ranger"]),
    multiple=True,
)
@click.option(
    "--lr",
    default=[0.0001],
    type=float,
    multiple=True,
)
@click.option(
    "--batch_size",
    "-bs",
    default=[16],
    type=int,
    multiple=True,
)
@click.option(
    "--frac_val",
    "-fv",
    default=[],
    type=float,
    multiple=True,
)
@click.option(
    "--k_fold",
    default=[],
    type=int,
    multiple=True,
)
@click.option(
    "--early_start",
    "-es",
    default=[100],
    type=int,
    multiple=True,
)
@click.option(
    "--epochs",
    "-ep",
    default=[100],
    type=int,
    multiple=True,
)
@click.option(
    "--output_tab",
    "-ot",
    default=[0],
    type=int,
    multiple=True,
)
@click.option("--double_img", is_flag=True, default=False, type=bool)
@click.option("--single_img", is_flag=True, default=False, type=bool)
@click.option("--return_command", is_flag=True, default=False, type=bool)
@click.option("--mem_avail", default=None, type=int)
def main(
    backbone: List[str],
    optim: List[str],
    lr: List[float],
    batch_size: List[int],
    frac_val: List[float],
    k_fold: List[int],
    early_start: List[int],
    epochs: List[int],
    output_tab: List[int],
    double_img: bool,
    single_img: bool,
    return_command: int,
    mem_avail: int,
):

    # Removing duplicated from lists
    backbone = list(dict.fromkeys(backbone))
    optim = list(dict.fromkeys(optim))
    lr = list(dict.fromkeys(lr))
    batch_size = list(dict.fromkeys(batch_size))
    frac_val = list(dict.fromkeys(frac_val))
    k_fold = list(dict.fromkeys(k_fold))
    early_start = list(dict.fromkeys(early_start))
    epochs = list(dict.fromkeys(epochs))
    output_tab = list(dict.fromkeys(output_tab))

    # Some checks
    if len(frac_val) != 0 and len(k_fold) != 0:
        print(
            "Use somente frac_val ou k_fold, os dois juntos não funciona",
            file=sys.stdout,
        )
        return

    if len(frac_val) == 0 and len(k_fold) == 0:
        frac_val = [0.2]

    params = {}
    params["backbone"] = backbone
    params["optim"] = optim
    params["lr"] = lr
    params["batch_size"] = batch_size
    if len(frac_val) != 0:
        params["frac_val"] = frac_val
    if len(k_fold) != 0:
        params["k_fold"] = k_fold
    params["early_start"] = early_start
    params["epochs"] = epochs
    if len(output_tab) != 0:
        params["output_tab"] = output_tab

    status_single = DataFrame()
    status_double = DataFrame()

    memory_usage = pd.read_csv(MODEL_USAGE_MEM)
    if single_img:
        status_single = list_status_models_tested(params, False, SUMMARY_PATH)
        status_single = status_single.merge(
            memory_usage, on=["backbone", "batch_size", "output_tab", "double_img"]
        )

    if double_img:
        status_double = list_status_models_tested(params, True, SUMMARY_PATH)
        status_double = status_double.merge(
            memory_usage, on=["backbone", "batch_size", "output_tab", "double_img"]
        )

    status = pd.concat([status_single, status_double])

    status = status.sort_values("mem_usage")
    status["cum_mem"] = status["mem_usage"].cumsum()
    status.reset_index(inplace=True, drop=True)

    if len(status) == 0:
        print("Stop", file=sys.stdout)
        return

    if return_command:
        if mem_avail:
            filtered_status = status[status["executed"] == 0]
            filtered_status = filtered_status[
                filtered_status["mem_usage"] <= mem_avail - SAFE_MEM_REDUCE
            ]
            commands = create_commands(filtered_status)
            command_selected = check_already_running(commands)
            if command_selected:
                print(command_selected, file=sys.stdout)
                return
            else:
                print("Full", file=sys.stdout)
                return
        else:
            print("Mem available not found", file=sys.stdout)
            return


if __name__ == "__main__":
    main()
