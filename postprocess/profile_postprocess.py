import sys
import pandas as pd


def combine_duplicates(df: pd.DataFrame):
    keys = ["Kernel", "Block"]
    cols_to_excl = ["Launches", "Grid", "Shared Mem Pipe %", "Shared Mem Pipe % std"] + keys
    value_cols = [c for c in df.columns if c not in cols_to_excl]

    weighted_sums = (
        df[value_cols]
        .mul(df["Launches"], axis=0)
        .groupby([df[k] for k in keys])
        .sum(min_count=1)
    )

    valid_weights = (
        df[value_cols].notna()
        .mul(df["Launches"], axis=0)
        .groupby([df[k] for k in keys])
        .sum(min_count=1)
    )

    means = weighted_sums.div(valid_weights)
    weight_totals = df.groupby(keys)["Launches"].sum()

    return weight_totals.to_frame().join(means).reset_index()


if __name__ == "__main__":
    combine_duplicates(pd.read_csv(sys.argv[1])).to_csv(sys.argv[2], index=False)
