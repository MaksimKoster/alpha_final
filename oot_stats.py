import pandas as pd
import os
import requests

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import sys
import importlib

from lightautoml2.automl.presets.tabular_presets import TabularAutoML
from lightautoml2.tasks import Task
import time
import gc

def fitting(path):
    try:
        current_data = os.listdir(path)
    except Exception:
        return "Не папка"
    else:
        current_data = os.listdir(path)

    start_time = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    N_THREADS = 16
    N_FOLDS = 5
    RANDOM_STATE = 42
    TIMEOUT = 1500
    TARGET_NAME = 'target'

    np.random.seed(RANDOM_STATE)
    torch.set_num_threads(N_THREADS)

    train_file = [data for data in current_data if data.endswith('train.parquet')][0]

    train_data = pd.read_parquet(path + f'/{train_file}')
    train_data = train_data.sort_values(by='id', ascending=True)

    new_train_size = len(train_data) - int(len(train_data) * 0.1)

    new_train = train_data.head(new_train_size)
    new_test = train_data.tail(len(train_data) - new_train_size)

    task = Task('binary')

    roles = {
        'target': TARGET_NAME,
        'drop': ['smpl','id']
    }

    automl = TabularAutoML(
        task = task,
        timeout = TIMEOUT,
        cpu_limit = N_THREADS,
        reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},
    )

    train_start_time = time.time()
    out_of_fold_predictions = automl.fit_predict(new_train, roles = roles, verbose = 1)
    train_end_time = time.time()

    predict_start_time = time.time()
    test_predictions = automl.predict(new_test)
    predict_end_time = time.time()

    oot_roc_auc = roc_auc_score(new_test[TARGET_NAME].values, test_predictions.data[:, 0])

    total_time = time.time() - start_time
    train_time = train_end_time - train_start_time
    predict_time = predict_end_time - predict_start_time

    return {
        "dataset": path,
        "train_size": len(new_train),
        "oot_size": len(new_test),
        'folds': N_FOLDS,
        "roc_auc_oot": oot_roc_auc,
        "time": total_time,
        "train_time": train_time,
        "predict_time": predict_time,
    }

def model():
    stats = {
        "dataset": [],
        "train_size": [],
        "oot_size": [],
        "roc_auc_oot": [],
        "time": [],
        "train_time": [],
        "predict_time": []
    }

    data = 'data'
    folders = os.listdir(data)

    for fold in folders:
        if fold == 'pd_fl' or fold == 'pd_ul_9':
            gc.collect()
            torch.cuda.empty_cache()

            data_path = data + f'/{fold}'

            prediction = fitting(path=data_path)

            stats['dataset'].append(str(fold))
            stats['train_size'].append(prediction['train_size'])
            stats['oot_size'].append(prediction['oot_size'])
            stats['folds'].append(prediction['folds'])
            stats['roc_auc_oot'].append(prediction['roc_auc_oot'])
            stats['time'].append(prediction['time'])
            stats['train_time'].append(prediction['train_time'])
            stats['predict_time'].append(prediction['predict_time'])

    ans = pd.DataFrame(stats)
    ans.to_csv('./light_stats_25m.csv')

if __name__ == "__main__":
    model()
