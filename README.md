<img width="226" alt="截屏2022-09-19 下午4 05 27" src="https://user-images.githubusercontent.com/108742116/190974653-4f7f8056-5bd4-4ece-bb06-4289b4989e11.png">

![](https://img.shields.io/badge/version-1.0-blue.svg) 
![](https://img.shields.io/badge/language-python-orange.svg)

****
Here is a repo for our submission *Heterogeneous Anomaly Detection for Software Systems via Semi-supervised Cross-modal Attention* for ICSE 2023.

## Architecture
![arch 001](https://user-images.githubusercontent.com/108742116/190979759-7e3ef203-3e1e-463b-9281-69b747b9486a.jpeg)

## Data
Our processed data are in the `data` directory and you may want to visit the raw data in `preprocess/raw_data`.

<details>
<summary>Sample</summary>
['edajgJf0': {'kpis': array([[-0.2446358, 0.40761607, ..., -0.11197388, -0.19489101], [-0.12524459, 0.34657824, ..., -0.11210638, -0.19490469], [ 0.04298847, 0.73634838, ..., -0.11170622, -0.19489621], ]]), 'logs': ['Registered signal handler for <*>', 'Registered signal handler for <*>', 'Registered signal handler for <*>', 'Changing <*> acls to: root', 'Changing <*> acls to: root', 'Changing <*> acls groups to:', 'Changing <*> acls groups to:', 'SecurityManager: authentication disabled; ui acls disabled; users with view permissions: Set(root); groups with view permissions: Set(); users with modify permissions: Set(root); groups with modify permissions: Set()', 'ApplicationAttemptId: <*>', 'Connecting to ResourceManager at slave1/172.17.0.3:8030', 'Registering the ApplicationMaster', 'Successfully created connection to <*> after <*> ms (0 ms spent in bootstraps)', 'Preparing Local resources', 'Will request 2 executor container(s), each with 4 core(s) and 4505 MB memory (including 409 MB of overhead)', 'Submitted 2 unlocalized container requests.', 'Started progress reporter thread with (heartbeat : 3000, initial allocation : 200) intervals', ...], 'kpi_label': 0, 'log_label': 0, 'label': 0},...]
</details>

## Tree
```
.
├── README.md
├── codes
│   ├── common
│   │   ├── __init__.py
│   │   ├── data_loads.py
│   │   ├── semantics.py
│   │   └── utils.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── fuse.py
│   │   ├── kpi_model.py
│   │   ├── log_model.py
│   │   └── utils.py
│   └── run.py
├── data
│   └── chunk_10_demo
│       ├── test.pkl
│       ├── train.pkl
│       └── unlabel.pkl
├── preprocess
│   ├── README.md
│   ├── get_chunks.py
│   ├── raw_data
│   │   ├── meta_info.json
│   │   └── raw_csvs
│   │       └── README.md
│   └── split_data.py
├── requirements.txt
├── result
│   └── final_res
│       ├── model.ckpt
│       ├── params.json
│       └── running.log
```

## Environment
We support python3.x $\geq$ 3.7. The environment can be built by:
```$ pip install -r requirements.txt```

## Result
The trained model and the final result records are in the `result` directory.

## Reproducing 
```$ cd codes && python run.py --data ../data/chunk_10_demo```

Note that we currently provide a demo dataset. We will release all the data after double-anonymous review. If you want the whole data, please leave a message with your contact information in Issues and we will contact you ASAP.

## Preview
![preview 001](https://user-images.githubusercontent.com/108742116/190979242-4d1024cc-4cac-476d-9a25-c6fd1a05be31.jpeg)

## TODO
* We will open up a website as a lightweight tool for users. 
