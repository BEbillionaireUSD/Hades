import json
meta_info = json.load("raw_data/meta_info.json")
workloads, faults, standards = meta_info["workloads"], meta_info["faults"], meta_info["standards"]

import numpy as np
def get_seq_chunk(chunks, seq_len=20, chunk_length=10):
    for k, item in chunks.items():
        logs = list(item["logs"])
        cur_len = len(logs)
        if cur_len > seq_len * chunk_length: # remove duplicate
            dup = set()
            del_idx = []
            for i, log in enumerate(logs):
                if log not in dup: dup.add(log)
                else: del_idx.append(i)
                if cur_len - len(del_idx) == seq_len * chunk_length: break
            new_logs = []
            for i, log in enumerate(logs):
                if i not in del_idx:
                    new_logs.append(log)
            logs = new_logs
            cur_len = len(logs)
            if cur_len > seq_len * chunk_length:
                if int(item["log_label"]) > 0: print(logs)
                logs = logs[:seq_len * chunk_length]
        elif cur_len < seq_len * chunk_length:
            logs += ["padding"]*(seq_len * chunk_length-cur_len)

        assert len(logs) == seq_len * chunk_length
        seqs = []
        for i in range(chunk_length):
            seqs.append(logs[i*seq_len: (i+1)*seq_len])
        chunks[k]["seqs"] = seqs
        chunks[k]["logs"] = logs
        for n,v in chunks[k].items():
            if not (isinstance(v, (np.ndarray, dict, list, int))):
                print(n, type(v))
                exit()
    return chunks
                
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--chunk_length", default=10, type=int)
params = vars(parser.parse_args())
chunk_length = params["chunk_length"]
dir = "../chunk_"+str(chunk_length)+"/"
os.makedirs(dir, exist_ok=True)

import os
import pickle

seen_wks = ["join", "lda", "pagerank", "web_pagerank", "nweight", "wordcount"]
seen_faults = ['net_delay', 'datanode_killed', 'datanode_suspended', 'mq_stress', 'slow_stress','resourcemanager_suspended','vm_stress']
train_chunks_labeled = {}
for wk in seen_wks:
    for task in seen_faults+standards:
        if os.path.exists(os.path.join("./inputs/pkls",wk, task+".pkl")):
            with open(os.path.join("./inputs/pkls",wk, task+".pkl"), "rb") as fr: 
                chunks = pickle.load(fr)
                train_chunks_labeled.update(chunks)

train_labels = [v['label'] for _, v in train_chunks_labeled.items()]
print("# train chunks: {}/{} ({:.4f}%)".format(sum(train_labels), len(train_labels), 100*(sum(train_labels)/len(train_labels))))
train_chunks = get_seq_chunk(train_chunks_labeled, 200 // chunk_length, chunk_length)
with open(dir+"train.pkl", "wb") as fw:
    pickle.dump(train_chunks, fw)
print("Train loaded")

test_wks = ['bayes','gbt','sort','pca','svd']
test_chunks = {}
for wk in test_wks:
    for task in faults+standards:
        if os.path.exists(os.path.join("./inputs/pkls", wk, task+".pkl")):
            with open(os.path.join("./inputs/pkls",wk, task+".pkl"), "rb") as fr: 
                chunks = pickle.load(fr)
                test_chunks.update(chunks)

test_labels = [v['label'] for _, v in test_chunks.items()]
print("# test chunks: {}/{} ({:.4f}%)".format(sum(test_labels), len(test_labels), 100*(sum(test_labels)/len(test_labels))))
test_chunks = get_seq_chunk(test_chunks, 200 // chunk_length, chunk_length)
with open(dir+"test.pkl", "wb") as fw:
    pickle.dump(test_chunks, fw)
print("Test loaded")

unlabel_chunks = {}
for wk in set(workloads)-set(seen_wks)-set(test_wks):
    for task in faults+standards:
        if os.path.exists(os.path.join("./inputs/pkls", wk, task+".pkl")):
            with open(os.path.join("./inputs/pkls",wk, task+".pkl"), "rb") as fr: 
                chunks = pickle.load(fr)
                unlabel_chunks.update(chunks)
for wk in seen_wks:
    for task in set(faults)-set(seen_faults):
        if os.path.exists(os.path.join("./inputs/pkls", wk, task+".pkl")):
            with open(os.path.join("./inputs/pkls",wk, task+".pkl"), "rb") as fr: 
                chunks = pickle.load(fr)
                unlabel_chunks.update(chunks)

unlabel_labels = [v['label'] for _, v in unlabel_chunks.items()]
print("# other chunks: {}/{} ({:.4f}%)".format(sum(unlabel_labels), len(unlabel_labels), 100*(sum(unlabel_labels)/len(unlabel_labels))))
unlabel_chunks = get_seq_chunk(unlabel_chunks, 200 // chunk_length, chunk_length)
with open(dir+"unlabel.pkl", "wb") as fw:
    pickle.dump(unlabel_chunks, fw)
print("Unlabel loaded")





