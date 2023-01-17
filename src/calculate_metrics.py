import pandas as pd
import numpy as np
import math
from sklearn.metrics import roc_auc_score, confusion_matrix
from params import SUMMARY_PATH, HISTORY, ROOT_DIR

VAL_BEST_AUC = "val_best_auc"


def calculate_oos(pred: pd.DataFrame):
    oos_pred = pred[pred["dataset"] == "oos"]
    sigmoid_outputs = oos_pred["pred_sigmoid"].values
    true = oos_pred["true"].values

    preds = sigmoid_outputs > 0.5

    epoch_auc = roc_auc_score(true, preds)
    tn, fp, fn, tp = confusion_matrix(true, preds).ravel()

    # Calculate specificity
    specificity = tn / (tn + fp)

    # Calculate sensitivity
    sensitivity = tp / (tp + fn)

    return epoch_auc, specificity, sensitivity


def assign(summary, model_id, column, value):
    summary.loc[summary["model_id"] == model_id, column] = value


def main():
    summary = pd.read_csv(SUMMARY_PATH)
    history = pd.read_csv(HISTORY)

    for _, row in summary.iterrows():
        if row["pred_oos"] == 1:
            model_id = row["model_id"]

            if not math.isnan(row["frac_val"]):
                file_name = "model_pred.csv"
                path = ROOT_DIR / "models" / row["model_id"] / file_name
                pred = pd.read_csv(path)
                epoch_auc, specificity, sensitivity = calculate_oos(pred)

                assign(summary, model_id, "oos_best_auc", epoch_auc)

                assign(summary, model_id, "max_sp_oos", specificity)

                assign(summary, model_id, "max_sn_oos", sensitivity)
            else:
                k_fold = int(row["k_fold"])
                list_auc = []
                list_specificity = []
                list_sensitivity = []

                for i in range(1, k_fold + 1):
                    file_name = "model_fold_" + str(i) + "_pred.csv"
                    path = ROOT_DIR / "models" / row["model_id"] / file_name
                    pred = pd.read_csv(path)
                    epoch_auc, specificity, sensitivity = calculate_oos(pred)
                    list_specificity.append(specificity)
                    list_sensitivity.append(sensitivity)
                    list_auc.append(epoch_auc)

                max_auc = np.max(list_auc)
                avg_auc = np.mean(list_auc)
                std_auc = np.std(list_auc)

                max_sp = np.max(list_specificity)
                avg_sp = np.mean(list_specificity)
                std_sp = np.std(list_specificity)

                max_sn = np.max(list_sensitivity)
                avg_sn = np.mean(list_sensitivity)
                std_sn = np.std(list_sensitivity)

                assign(summary, model_id, "oos_best_auc", max_auc)
                assign(summary, model_id, "avg_best_auc_oos", avg_auc)
                assign(summary, model_id, "std_oos_auc", std_auc)

                assign(summary, model_id, "max_sp_oos", max_sp)
                assign(summary, model_id, "avg_sp_oos", avg_sp)
                assign(summary, model_id, "std_sp_oos", std_sp)

                assign(summary, model_id, "max_sn_oos", max_sn)
                assign(summary, model_id, "avg_sn_oos", avg_sn)
                assign(summary, model_id, "std_sn_oos", std_sn)

        if row["history_added"] == 1:
            model_id = row["model_id"]

            if not math.isnan(row["frac_val"]):
                history_filtered = history[history["model_id"] == model_id]
                index_max = history_filtered["val_auc_history"].idxmax()
                sp = history.loc[index_max, "val_specificity_history"]
                sn = history.loc[index_max, "val_sensitivity_history"]

                assign(summary, model_id, "idx_max_sp_val", sp)
                assign(summary, model_id, "idx_max_sn_val", sn)
            else:
                k_fold = int(row["k_fold"])
                list_auc = []
                list_specificity = []
                list_sensitivity = []

                for i in range(1, k_fold + 1):
                    history_filtered = history[
                        (history["model_id"] == model_id) & (history["iter"] == i)
                    ]
                    max_auc = history_filtered["val_auc_history"].max()
                    index_max = history_filtered["val_auc_history"].idxmax()
                    list_auc.append(max_auc)

                    sp = history.loc[index_max, "val_specificity_history"]
                    sn = history.loc[index_max, "val_sensitivity_history"]
                    list_specificity.append(sp)
                    list_sensitivity.append(sn)

                    max_auc = np.max(list_auc)
                    avg_auc = np.mean(list_auc)
                    std_auc = np.std(list_auc)

                    max_sp = np.max(list_specificity)
                    avg_sp = np.mean(list_specificity)
                    std_sp = np.std(list_specificity)

                    max_sn = np.max(list_sensitivity)
                    avg_sn = np.mean(list_sensitivity)
                    std_sn = np.std(list_sensitivity)

                    assign(summary, model_id, "avg_best_auc_val", avg_auc)
                    assign(summary, model_id, "std_val_auc", std_auc)

                    assign(summary, model_id, "idx_max_sp_val", max_sp)
                    assign(summary, model_id, "idx_avg_sp_val", avg_sp)
                    assign(summary, model_id, "idx_std_sp_val", std_sp)

                    assign(summary, model_id, "idx_max_sn_val", max_sn)
                    assign(summary, model_id, "idx_avg_sn_val", avg_sn)
                    assign(summary, model_id, "idx_std_sn_val", std_sn)

    summary.to_csv(SUMMARY_PATH, index=False)

if __name__ == "__main__":
    main()
