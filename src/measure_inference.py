# TODO: adding time cross validation and total, datetime.now, and torch/torchvision version. Add
# shallow copy to double img shared parameters model.
#!/usr/bin/env python
import torch
import click
import random
import pandas as pd
import numpy as np


from model import init_model
from dotenv import load_dotenv

load_dotenv()


# Fix seed's for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


@click.command
@click.option(
    "--model_id",
    required=True,
    type=str,
    help="Model id to measure inference time",
)
def main(model_id):
    summary = pd.read_csv("../model_summary.csv", sep=",")
    summary: pd.DataFrame = summary.drop_duplicates(subset=["model_id"])
    data = pd.read_csv("../data.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numerical_columns = data.select_dtypes(include=np.number).columns.tolist()
    numerical_columns.remove("label")
    ft_size = len(numerical_columns)
    print(f"Testing model...")
    print(model_id)
    print(summary.shape)
    row = summary[summary["model_id"] == model_id]
    print(row.squeeze())
    double_img = True if row["double_img"].values[0] > 0 else False

    backbone = row["backbone"].values[0]
    pretrained = False

    output_tab = (
        int(row["output_tab"].values[0]) if row["output_tab"].values[0] > 0 else None
    )

    print(output_tab)
    model, input_size = init_model(
        backbone,
        pretrained,
        double_img,
        output_tab,
        ft_size,
    )
    model.to(device)
    model.eval()
    dummy_input = torch.ones(2, 3, input_size, input_size, dtype=torch.float).to(device)
    ft_dummy = torch.ones(2, ft_size, dtype=torch.float).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        if double_img and not output_tab:
            _ = model(dummy_input, dummy_input, None)
        elif double_img and output_tab:
            _ = model(dummy_input, dummy_input, ft_dummy)
        elif not double_img and output_tab:
            _ = model(dummy_input, None, ft_dummy)
        else:
            _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            if double_img and not output_tab:
                _ = model(dummy_input, dummy_input, None)
            elif double_img and output_tab:
                _ = model(dummy_input, dummy_input, ft_dummy)
            elif not double_img and output_tab:
                _ = model(dummy_input, None, ft_dummy)
            else:
                _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)


if __name__ == "__main__":
    main()
