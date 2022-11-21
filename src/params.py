from pathlib import Path
import os

# PARAMS
ROOT_DIR = Path(os.path.abspath(__file__)).parents[1].absolute()
SUMMARY_PATH = ROOT_DIR / "summary.csv"
MODEL_USAGE_MEM = ROOT_DIR / "usage_mem_gpu.csv"
BACKBONE_ARG = "--backbone"
BATCHSIZE_ARG = "--batch_size"
DOUBLE_IMG_ARG = "--double_img"
OUTPUT_TAB_ARG = "--output_tab"
EARLY_START_ARG = "--early_start"
OPTIM_ARG = "--optim"
FRAC_VAL_ARG = "--frac_val"
K_FOLD_ARG = "--k_fold"
LR_ARG = "--lr"
EPOCHS_ARG = "--epochs"
