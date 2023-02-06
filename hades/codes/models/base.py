import os
import time
import copy
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, recall_score, precision_score
import logging
import warnings
warnings.filterwarnings("ignore", module="sklearn")

import numpy as np
from models.fuse import MultiModel
from models.log_model import LogModel
from models.kpi_model import KpiModel

class pseudo_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class BaseModel(nn.Module):
    def __init__(self, device, var_nums, vocab_size=300, data_type="fuse", **kwargs):
        super(BaseModel, self).__init__()
        # Training
        [self.epoches_1, self.epoches_2] = kwargs["epoches"]
        self.confidence = kwargs["confidence"]
        self.alpha = kwargs["alpha"]
        self.batch_size = kwargs["batch_size"]
        self.learning_rate = kwargs["learning_rate"]
        self.patience = kwargs["patience"] # > 0: use early stop
        self.device = device
        self.var_nums = var_nums

        self.model_save_dir = os.path.join(kwargs["result_dir"], kwargs["hash_id"])
        self.model_save_file = os.path.join(self.model_save_dir, 'model.ckpt')
        if data_type == "fuse": 
            self.model = MultiModel(var_nums=var_nums, vocab_size=vocab_size, device=device, **kwargs)
        elif data_type == "log": 
            self.model = LogModel(vocab_size=vocab_size, device=device, **kwargs)
        elif data_type == "kpi": 
            self.model = KpiModel(var_nums=var_nums, device=device, **kwargs)
        self.model.to(device)
        self.train_time = []
    
    def __input2device(self, batch_input):
        res = {}
        for key, value in batch_input.items():
            if isinstance(value, list):
                res[key] = [v.to(self.device) for v in value]
            else:
                res[key] = value.to(self.device) 
        return res
    
    def load_model(self, model_save_file=""):
        self.model.load_state_dict(torch.load(model_save_file, map_location=self.device))

    def save_model(self, state):
        try:
            torch.save(state, self.model_save_file, _use_new_zipfile_serialization=False)
        except:
            torch.save(state, self.model_save_file)
    
    def inference(self, data_loader):
        self.model.eval()
        data = []
        inference_time = []
        with torch.no_grad():
            for _input in data_loader:
                infer_start = time.time()
                result = self.model.forward(self.__input2device(_input), flag=True)
                inference_time.append(time.time() - infer_start)
                if result["conf"][0] >= self.confidence:
                    sample = {
                        "idx": _input["idx"].item(),
                        "label": int(result["y_pred"][0]),
                        "kpi_features": [ts.squeeze() for ts in _input["kpi_features"]],
                        "log_features": _input["log_features"].squeeze()
                    }
                    data.append(sample)
        logging.info("Inference delay {:.4f}".format(np.mean(inference_time)))
        return pseudo_Dataset(data)

    def evaluate(self, test_loader, datatype="Test"):
        self.model.eval()
        res = defaultdict(list)
        
        with torch.no_grad():
            batch_cnt = 0 
            for batch_input in test_loader:
                batch_cnt += 1
                result = self.model.forward(self.__input2device(batch_input), flag=True)
                res["pred"].extend(result["y_pred"].tolist())
                res["true"].extend(batch_input["label"].data.cpu().numpy().tolist())
                res["idx"].extend(batch_input["idx"].data.cpu().numpy().tolist())
    
        eval_results = {
                "f1": f1_score(res["true"], res["pred"]),
                "rc": recall_score(res["true"], res["pred"]),
                "pc": precision_score(res["true"], res["pred"]),
            }
        logging.info("{} -- {}".format(datatype, ",".join([k+":"+str(f"{v:.4f}") for k, v in eval_results.items()])))

        return eval_results
    
    def supervised_fit(self, train_loader, test_loader):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_f1 = -1
        best_state, best_test_scores = None, None

        pre_loss, worse_count = float("inf"), 0
        for epoch in range(1, self.epoches_1+1):
            self.model.train()
            batch_cnt, epoch_loss = 0, 0.0
            epoch_time_start = time.time()
            for batch_input in train_loader:
                optimizer.zero_grad()
                loss = self.model.forward(self.__input2device(batch_input))["loss"]
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_cnt += 1
            epoch_time_elapsed = time.time() - epoch_time_start
            epoch_loss = epoch_loss / batch_cnt

            self.train_time.append(epoch_time_elapsed)
            logging.info("Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, self.epoches_1, epoch_loss, epoch_time_elapsed))

            #train_results = self.evaluate(train_loader, datatype="Train")
            test_results = self.evaluate(test_loader, datatype="Test")

            if test_results["f1"] > best_f1:
                best_f1 = test_results["f1"]
                best_test_scores = test_results
                best_state = copy.deepcopy(self.model.state_dict())

        self.save_model(best_state)
        self.load_model(self.model_save_file)
        test_results = self.evaluate(test_loader, datatype="Test")
        logging.info("*** Test F1 {:.4f}  of supervised traning".format(test_results["f1"]))

        return best_test_scores

    def fit(self, train_loader, unlabel_loader, test_loader):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_f1 = -1
        best_state, best_test_scores = None, None
        
        ##############################################################################
        ####                    Training with labeled data                        ####
        ##############################################################################
        pre_loss, worse_count = float("inf"), 0
        for epoch in range(1, self.epoches_1+1):
            self.model.train()
            batch_cnt, epoch_loss = 0, 0.0
            epoch_time_start = time.time()
            for batch_input in train_loader:
                optimizer.zero_grad()
                loss = self.model.forward(self.__input2device(batch_input))["loss"]
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_cnt += 1
            epoch_time_elapsed = time.time() - epoch_time_start
            epoch_loss = epoch_loss / batch_cnt

            self.train_time.append(epoch_time_elapsed)
            logging.info("Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, self.epoches_1, epoch_loss, epoch_time_elapsed))

            train_results = self.evaluate(train_loader, datatype="Train")
            test_results = self.evaluate(test_loader, datatype="Test")

            if test_results["f1"] > best_f1:
                best_f1 = test_results["f1"]
                best_test_scores = test_results
                best_state = copy.deepcopy(self.model.state_dict())

        self.save_model(best_state)
        self.load_model(self.model_save_file)
        test_results = self.evaluate(test_loader, datatype="Test")
        logging.info("*** Test F1 {:.4f}  of traning phase 1".format(test_results["f1"]))

        ##############################################################################
        ####              Training with labeled data and pseudo data              ####
        ##############################################################################
        pseudo_data = self.inference(unlabel_loader)
        pseudo_loader = DataLoader(pseudo_data, batch_size=self.batch_size, shuffle=True)
        
        pre_loss, worse_count = float("inf"), 0
        phase = False
        for epoch in range(1, self.epoches_2):
            self.model.train()
            batch_cnt, epoch_loss = 0, 0.0
            epoch_time_start = time.time()
            
            train_iterator = iter(train_loader)
            for pseudo_input in pseudo_loader:
                try:
                    train_input = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    train_input = next(train_iterator)

                optimizer.zero_grad()
                loss_1 = self.model.forward(self.__input2device(train_input))["loss"]
                loss_2 = self.model.forward(self.__input2device(pseudo_input))["loss"]
                loss = (1-self.alpha)*loss_1+self.alpha*loss_2
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_cnt += 1
            
            epoch_time_elapsed = time.time() - epoch_time_start
            epoch_loss = epoch_loss / batch_cnt
            self.train_time.append(epoch_time_elapsed)
            logging.info("Epoch {}/{}, training loss with real & pseudo labels: {:.5f} [{:.2f}s]".format(epoch, self.epoches_2, epoch_loss, epoch_time_elapsed))

            test_results = self.evaluate(test_loader, datatype="Test")

            if test_results["f1"] > best_f1:
                best_f1 = test_results["f1"]
                best_test_scores = test_results
                best_state = copy.deepcopy(self.model.state_dict())
                phase = True
            
            if epoch_loss > pre_loss:
                worse_count += 1
                if self.patience > 0 and worse_count >= self.patience:
                    logging.info("Early stop at epoch: {}".format(epoch))
                    break
            else: worse_count = 0
            pre_loss = epoch_loss
        
        self.save_model(best_state)
        self.load_model(self.model_save_file)
        test_results = self.evaluate(test_loader, datatype="Test")
        if phase:
            logging.info("*** Test F1 {:.4f} of traning phase 2".format(test_results["f1"]))
        else:
            logging.info("---- Training Phase 2 has no benifit!")
        
        logging.info("Best f1: test f1 {:.4f}".format(best_f1))
        return best_test_scores
