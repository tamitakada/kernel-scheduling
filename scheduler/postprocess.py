import sys
import os

import pandas as pd 
import numpy as np


if __name__ == "__main__":
    f1 = sys.argv[1]
    f2 = sys.argv[2]

    df1 = pd.read_csv(f1, sep="\t")[-200000:]
    df2 = pd.read_csv(f2, sep="\t")
    
    print(len(df1), len(df2))
    
    num_batches = [0, 0]
    
    for i,s in enumerate(df1["start_ns"].unique()):
        print(i)
    
        bsize = 0
        bsize += len(df1[df1["start_ns"]==s])
        assert(bsize == 1)
        
        bsize += len(df2[df2["start_ns"]==s])
        assert(bsize <= 2)
        
        num_batches[bsize - 1] += 1
        
    for i,s in enumerate(df2["start_ns"].unique()):
        print(i)
    
        bsize = 0
        bsize += len(df2[df2["start_ns"]==s])
        assert(bsize == 1)
        
        if df1[df1["start_ns"]==s].empty:
            num_batches[bsize - 1] += 1
            
    print(num_batches)
    
#    total_exec_s1 = (df1[df1["hook_tag"] == "unknown"]["kernel_us"].sum() + \
#        df1[df1["hook_tag"] != "unknown"]["kernel_us"].sum() + \
#        df1[df1["hook_tag"] != "unknown"]["prefetch_us"].sum()) / (10**6)
#    total_exec_s2 = df2["kernel_us"].sum() / (10**6)
#    ideal_exec_s1 = df1["kernel_us"].sum() / (10**6)
#    
#    print(f"Real execs: {total_exec_s1}, {total_exec_s2}, diff: {total_exec_s1 - total_exec_s2}")
#    print(f"Ideal exec: {ideal_exec_s1}, diff: {ideal_exec_s1 - total_exec_s2}")
#    print(f"Real TPUT: {5 / total_exec_s1}, {5 / total_exec_s2}, {5 / ideal_exec_s1}")
#    print()
#    
#    df1 = df1[df1["hook_tag"] != "unknown"]
#    
#    df2 = df2[df2["hook_tag"] != "unknown"]
#
#    kernel_types = df1["kernel_name"].unique()
#    for kt in df1.groupby("kernel_name")["kernel_us"].mean().sort_values().index.tolist():
#        df11 = df1[(df1["kernel_name"]==kt) & (df1["prefetch_us"] > 0)]
#        df21 = df2[df2["kernel_name"]==kt]
#        
#        if len(df11) == 0:
#            continue
#        
#        print(kt)
#        print(f"With #: {len(df11)}, Without #: {len(df21)}")
#        print(df21["kernel_us"].mean(), df11["kernel_us"].mean() - df21["kernel_us"].mean())
#        print()
#        
#    print()
