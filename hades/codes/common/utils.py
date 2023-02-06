import os
import json
from datetime import datetime, timedelta
import numpy as np
def dump_scores(result_dir, hash_id, scores, train_time):
    with open(os.path.join(result_dir, hash_id, "info_score.txt"), "w") as fw: #Details
        fw.write('Experiment '+hash_id+': '+(datetime.now()+timedelta(hours=8)).strftime("%Y/%m/%d-%H:%M:%S")+'\n')
        fw.write('Train time/epoch {:.4f}\n'.format(np.mean(train_time)))
        fw.write("* Test -- " + '\t'.join(["{}:{:.4f}".format(k, v) for k,v in scores.items()])+'\n\n')
        
def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(obj, fw, sort_keys=True, indent=4,
            separators=(",", ": "), ensure_ascii=False,)

import hashlib
import logging
def dump_params(params):
    hash_id = hashlib.md5(str(sorted([(k, v) for k, v in params.items()])).encode("utf-8")).hexdigest()[0:8]
    result_dir = os.path.join(params["result_dir"], hash_id)
    os.makedirs(result_dir, exist_ok=True)

    json_pretty_dump(params, os.path.join(result_dir, "params.json"))

    log_file = os.path.join(result_dir, "running.log")
    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    #logging.info(json.dumps(params, indent=4))
    return hash_id

import random
def decision(probability):
    return random.random() < probability
    
import torch
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

import pickle
def dump_pickle(obj, file_path):
    logging.info("Dumping to {}".format(file_path))
    with open(file_path, "wb") as fw:
        pickle.dump(obj, fw)


def load_pickle(file_path):
    logging.info("Loading from {}".format(file_path))
    with open(file_path, "rb") as fr:
        return pickle.load(fr)