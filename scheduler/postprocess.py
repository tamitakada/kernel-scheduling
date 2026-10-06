import sys
import os

import pandas as pd 
import numpy as np


if __name__ == "__main__":
    f1 = sys.argv[1]
    f2 = sys.argv[2]

    df1 = pd.read_csv(f1, sep="\t")
    df2 = pd.read_csv(f2, sep="\t")

    kernel_types = df1["kernel_name"].unique()
    for kt in df1.groupby("kernel_name")["kernel_us"].mean().sort_values().index.tolist():
        print(kt)

        df11 = df1[df1["kernel_name"]==kt]
        df21 = df2[df2["kernel_name"]==kt]

        print(df11["kernel_us"].mean() - df21["kernel_us"].mean())

        # mean_qtime_diff = df11["queueing_time"].mean() - df21["queueing_time"].mean()
        # mean_etime_diff = df11["exec_time"].mean() - df21["exec_time"].mean()

        # print(df11["exec_time"].mean(), mean_qtime_diff, mean_etime_diff)
        print()