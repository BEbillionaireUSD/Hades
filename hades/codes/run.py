from torch.utils.data import DataLoader
from common.data_loads import load_sessions, Process
from common.utils import *
import torch
import logging
from models.base import BaseModel

import argparse
parser = argparse.ArgumentParser()
### Model params
parser.add_argument("--supervised", action="store_true")
parser.add_argument("--gpu", default=True, type=lambda x: x.lower() == "true")
parser.add_argument("--epoches", default=[50, 50], type=int, nargs='+')
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--confidence", default=0.92, type=float)
parser.add_argument("--alpha", default=0.5, type=float)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--patience", default=10, type=int)
parser.add_argument("--random_seed", default=42, type=int)
parser.add_argument("--optim", default=-1, type=float)
parser.add_argument("--weight_decay", default=0, type=float)

##### Munual params
parser.add_argument("--window_size", default=10, type=int)
parser.add_argument("--hidden_size", default=128, type=int, help="Dim of the commnon feature space")
parser.add_argument("--self_attention", default=True, type=lambda x: x.lower() == "true")
# Fuse params
parser.add_argument("--linear_size", default=512, type=int, help="hidden size of the decoder")
parser.add_argument("--decoder_layer_num", default=6, type=int)
parser.add_argument("--data_type", default="fuse", choices=["fuse", "log", "kpi"])
parser.add_argument("--fuse_type", default="cross_attn", choices=["concat", "cross_attn", "sep_attn"])
parser.add_argument("--attention_type", default="general", choices=["general", "dot"])

### Kpi params
parser.add_argument("--kpi_architect", default="by_aspect", type=str, choices=["by_aspect", "by_metric"])
parser.add_argument("--temporal_kernel_sizes", default=[2, 2], type=int, nargs='+')
parser.add_argument("--temporal_hidden_sizes", default=[64, 4], type=int, nargs='+')
parser.add_argument("--temporal_dropout", default=0, type=float)
parser.add_argument("--pooling", default=True, type=lambda x: x.lower() == "true")
# Inner params
parser.add_argument("--inner_hidden_sizes", default=[256, 256, 256], type=int, nargs='+')
parser.add_argument("--inner_kernel_sizes", default=[3, 3, 3], type=int, nargs='+')
parser.add_argument("--inner_dropout", default=0.5, type=float)

### Log params
parser.add_argument("--feature_type", default="word2vec", type=str, choices=["word2vec", "sequential"])
parser.add_argument("--log_window_size", default=20, type=int)
parser.add_argument("--log_layer_num", default=4, type=int)
parser.add_argument("--log_dropout", default=0.1, type=float)
parser.add_argument("--transformer_hidden", default=1024, type=int)

# Word params
parser.add_argument("--word2vec_model_type", default="fasttext", type=str, choices=["naive","fasttext","skip-gram"])
parser.add_argument("--word_embedding_dim", default=32, type=int)
parser.add_argument("--word_window", default=5, type=int)
parser.add_argument("--word2vec_epoch", default=50, type=int)

### Control params
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--pre_model", default=None, type=str)
parser.add_argument("--word2vec_save_dir", default="../trained_wv/", type=str)
parser.add_argument("--result_dir", default="../result/", type=str)
parser.add_argument("--main_model", default="hades", choices=["hades", "join-hades", "concat-hades", "sep-hades", "agn-hades", "one-hades", "met-hades", "anno-hades"])

params = vars(parser.parse_args())
params["hash_id"] = dump_params(params)
seed_everything(params["random_seed"])

###### Ablation Experiments #######
if params["main_model"] == "join-hades":
    params["fuse_type"] = "join"
elif params["main_model"] == "concat-hades":
    params["fuse_type"] = "concat"
elif params["main_model"] == "sep-hades":
    params["fuse_type"] = "sep_attn"
elif params["main_model"] == "agn-hades":
    params["kpi_architect"] = "by_metric"
elif params["main_model"] == "one-hades":
    params["feature_type"] == "sequential"
elif params["main_model"] == "met-hades":
    params["data_type"] == "sequential"
elif params["main_model"] == "anno-hades":
    params["supervised"] == "True"

if params["gpu"] and torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info("Using GPU...")
else:
    device = torch.device("cpu")
    logging.info("Using CPU...")

def main(var_nums):
    logging.info("^^^^^^^^^^ Current Model:"+params["main_model"]+", "+str(params["hash_id"])+" ^^^^^^^^^^")

    ###### Load data  ######
    train_chunks, unlabel_chunks, test_chunks = load_sessions(data_dir=params["data"])

    if params["supervised"]:
        train_chunks.update(unlabel_chunks)
    processed = Process(var_nums, train_chunks, unlabel_chunks, test_chunks, **params)

    bz = params["batch_size"]
    train_loader = DataLoader(processed.dataset["train"], batch_size=bz, shuffle=True, pin_memory=True)
    unlabel_loader = DataLoader(processed.dataset["unlabel"], shuffle=True, pin_memory=True)
    test_loader = DataLoader(processed.dataset["test"], batch_size=bz, shuffle=False, pin_memory=True)

    ##### Build/Train model #####
    if params['data_type'] != 'kpi':
        vocab_size = processed.ext.meta_data["vocab_size"]
        logging.info("Known word number: {}".format(vocab_size))
    else: vocab_size=300
    model = BaseModel(device=device, var_nums=var_nums, vocab_size=vocab_size, **params)

    if params["pre_model"] is None: #train
        if params["supervised"]:
            scores = model.supervised_fit(train_loader, test_loader)
        else:
            scores = model.fit(train_loader, unlabel_loader, test_loader)
    else:
        model.load_model(params["pre_model"])
        scores = model.evaluate(test_loader)
    
    ##### Record results #####
    dump_scores(params["result_dir"], params["hash_id"], scores, model.train_time)
    logging.info("Current hash id {}".format(params["hash_id"]))

if __name__ == "__main__":
    main([4,3,2,2])

        
