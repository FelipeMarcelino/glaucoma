import sys
import pandas as pd
import click
import pickle

from params import SUMMARY_PATH, OCT_PRESENCE, DUAL_IMAGE
from utils import open_result_file


def extract_history(summary, history_df):

    history_df_list = []
    for _, row in summary.iterrows():
        model_id = row["model_id"]
        if isinstance(history_df, pd.DataFrame):
            test_df = history_df[history_df["model_id"] == model_id]
            if len(test_df) > 0:
                continue
        try:
            f = open_result_file(model_id)
            if not f:
                continue
            dump = pickle.load(f)
            list_of_pandas = []
            for key, value in dump.items():
                list_of_pandas_column = []
                for idx in range(len(value)):
                    df_tmp = pd.DataFrame(columns=[key, "iter", "epochs"])
                    df_tmp[key] = value[idx]
                    df_tmp["epochs"] = list(range(1, len(value[idx]) + 1))
                    df_tmp["iter"] = idx + 1
                    list_of_pandas_column.append(df_tmp)
                df_tmp_column = pd.concat(list_of_pandas_column)
                df_tmp_column.reset_index(inplace=True, drop=True)
                list_of_pandas.append(df_tmp_column)

            df_results = pd.concat(list_of_pandas, join="inner", axis=1)
            df_results = df_results.loc[:, ~df_results.columns.duplicated()].copy()
            df_results = df_results.reindex(sorted(df_results.columns), axis=1)
            df_results["backbone"] = row["backbone"]
            df_results[OCT_PRESENCE] = row[OCT_PRESENCE]
            df_results[DUAL_IMAGE] = row[DUAL_IMAGE]
            df_results["optim"] = row["optim"]
            df_results["lr"] = row["lr"]
            df_results["frac_val"] = row["frac_val"]
            df_results["k_fold"] = row["k_fold"]
            df_results["model_id"] = row["model_id"]
            history_df_list.append(df_results)
        except FileNotFoundError:
            continue

    return history_df_list


@click.command
@click.option(
    "-hf", "--history_file", default="../history.csv", type=click.Path(exists=False)
)
def main(history_file):

    try:
        summary = pd.read_csv(SUMMARY_PATH)
    except FileNotFoundError as e:
        print("Summary file does not exist!", file=sys.stderr)
        print(f"Exception: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        history_df = pd.read_csv(history_file)
    except FileNotFoundError:
        history_df = None

    history_df_list = extract_history(summary, history_df)

    if len(history_df_list) > 0:
        history_df_tmp = pd.concat(history_df_list)
        if isinstance(history_df, pd.DataFrame):
            history_df_final = pd.concat([history_df, history_df_tmp])
        else:
            history_df_final = history_df_tmp
        history_df_final.to_csv(history_file, index=False)
        model_ids = history_df_final["model_id"].values.tolist()
        summary.loc[summary["model_id"].isin(model_ids), "history_added"] = 1
        summary.to_csv(SUMMARY_PATH, index=False)
    else:
        print("There is no history to update!")


if __name__ == "__main__":
    main()
