<img width="226" alt="截屏2022-09-19 下午4 05 27" src="https://user-images.githubusercontent.com/108742116/190974653-4f7f8056-5bd4-4ece-bb06-4289b4989e11.png">

![](https://img.shields.io/badge/version-1.0-blue.svg) 
![](https://img.shields.io/badge/language-python-orange.svg)

****
Here is a repo for our paper published at ICSE 2023: *Heterogeneous Anomaly Detection for Software Systems via Semi-supervised Cross-modal Attention*.

## Data
Our data are at https://doi.org/10.5281/zenodo.7609780.

## Environment
We support python3.x $\geq$ 3.7. The environment can be built by:
```$ pip install -r requirements.txt```

## Reproducing 
```$ cd codes && python run.py --data ../data/chunk_10```

## Architecture
![arch 001](https://user-images.githubusercontent.com/108742116/190979759-7e3ef203-3e1e-463b-9281-69b747b9486a.jpeg)

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
│   └── chunk_10
│       ├── test.pkl
│       ├── train.pkl
│       └── unlabel.pkl
├── preprocess
│   ├── README.md
│   ├── get_chunks.py
│   └── split_data.py
├── requirements.txt
```

## Result
The trained model and the final result records are in the `result` directory.

## Preview
![preview 001](https://user-images.githubusercontent.com/108742116/190979242-4d1024cc-4cac-476d-9a25-c6fd1a05be31.jpeg)
