import os
import pickle
import logging
import numpy as np
from torch.utils.data import Dataset
from common.semantics import FeatureExtractor
import logging

def load_sessions(data_dir): #both log and kpi
    logging.info("Load from {}".format(data_dir))
    with open(os.path.join(data_dir, "train.pkl"), "rb") as fr:
        train = pickle.load(fr)
    with open(os.path.join(data_dir, "unlabel.pkl"), "rb") as fr:
        unlabel = pickle.load(fr)
    with open(os.path.join(data_dir, "test.pkl"), "rb") as fr:
        test = pickle.load(fr)
    return train, unlabel, test

class myDataset(Dataset):
    def __init__(self, sessions):
        self.data = []
        self.idx2id = {}
        for idx, block_id in enumerate(sessions.keys()):
            self.idx2id[idx] = block_id
            item = sessions[block_id]
            sample = {
                'idx': idx,
                'label': int(item['label']),
                'kpi_features': item['kpi_features'],
                'log_features': item['log_features']
            }
            self.data.append(sample)
                
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    def __get_session_id__(self, idx):
        return self.idx2id[idx]

class Process():
    def __init__(self, var_nums, labeled_train, unlabel_train, test_chunks, supervised=False, **kwargs):
        self.var_nums = var_nums

        self.ext = FeatureExtractor(**kwargs)
        self.__train_ext(labeled_train, unlabel_train)
        
        labeled_train = self.ext.transform(labeled_train)
        test_chunks = self.ext.transform(test_chunks, datatype="test")
        labeled_train = self.__transform_kpi(labeled_train)
        test_chunks = self.__transform_kpi(test_chunks)

        if not supervised:
            unlabel_train = self.ext.transform(unlabel_train, datatype="unlabel train")
            unlabel_train = self.__transform_kpi(unlabel_train)

        logging.info('Data loaded done!')
        
        
        self.dataset = {
            'train': myDataset(labeled_train),
            'unlabel': myDataset(unlabel_train) if not supervised else None,
            'test':  myDataset(test_chunks),
        }
        
    def __train_ext(self, a, b):
        a.update(b)
        self.ext.fit(a)
    
    def __transform_kpi(self, chunks):
        for id, dict in chunks.items():
            kpis = dict['kpis']
            if kpis.shape[0] != sum(self.var_nums): kpis = kpis.T
            chunks[id]['kpi_features'] = []
            pre_num = 0
            for num in self.var_nums:
                chunks[id]['kpi_features'].append(kpis[pre_num:pre_num+num, :])
                pre_num += num
        return chunks