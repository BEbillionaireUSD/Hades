import json
meta_info = json.load("raw_data/meta_info.json")
workloads, faults, standards, metrics = meta_info["workloads"], meta_info["faults"], meta_info["standards"], meta_info["metrics"]

import string
import random
chunkids = set()
src = string.ascii_letters + string.digits
def get_chunkid():
    while(True):
        chunkid = random.sample(src, 8)
        random.shuffle(chunkid)
        chunkid = ''.join(chunkid)
        #print(chunkid)
        if chunkid not in chunkids:
            chunkids.add(chunkid)
            return chunkid

def get_chunk_times(timestamps, chunk_length=10, step=1):
    times = []
    s, e = timestamps.min(), timestamps.max()
    timestamps = timestamps.values
    i = s+chunk_length
    while i<=e+1:
        times.append((i-chunk_length, i))
        i += step
    return times

import pandas as pd
def get_chunk(wk, task, chunk_length=10, **kwargs):
    kpi_data = pd.read_csv("./raw_data/wk_task_data/"+wk+'/'+task+'/trans_kpi.csv')
    log_data = pd.read_csv("./raw_data/wk_task_data/"+wk+'/'+task+'/trans_log.csv')
    timestamps = kpi_data["timestamp"]
    chunk_times = get_chunk_times(timestamps, chunk_length=chunk_length, **kwargs)
    chunks = {}
    for (s,e) in chunk_times:
        kpi_rows = kpi_data[(kpi_data['timestamp']>=s) & (kpi_data['timestamp']<e)]
        log_rows = log_data[(log_data['Timestamp']>=s) & (log_data['Timestamp']<e)]
        kpis = kpi_rows[kpi_rows.columns[2:]].to_numpy()
        kpi_label = kpi_rows["label"].sum() > 0
        log_label = 0

        if not log_rows.empty:
            logs = log_rows["DrainTemplate"].values
            log_label = log_rows["Label"].sum() > 0
        else: 
            logs = ["padding"]  
        chunk_id = get_chunkid()
        chunks[chunk_id] = {"kpis":kpis, "logs":logs, "kpi_label": int(kpi_label), "log_label": int(log_label)}
        chunks[chunk_id]["label"] = int(log_label | kpi_label)
    return chunks
        
import os
import pickle
def run(**kwargs):
    for wk in workloads:
        aim_dir = "./inputs/pkls/"+wk+'/'
        os.makedirs(aim_dir, exist_ok=True)
        for task in faults+standards:
            if not os.path.exists("./raw_data/wk_task_data/"+wk+'/'+task): 
                continue
            chunks = get_chunk(wk, task, **kwargs)
            aim_path = aim_dir+task+'.pkl'
            if os.path.exists(aim_path): 
                os.remove(aim_path)
            with open(aim_path, "wb") as fw:
                pickle.dump(chunks, fw)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--chunk_length", default=10, type=int)
parser.add_argument("--step", default=1, type=int)
params = vars(parser.parse_args())
if "__main__" == __name__:
    run(**params)

    



