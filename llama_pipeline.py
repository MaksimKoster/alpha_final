import pandas as pd
import os
import requests

import numpy as np
import pandas as pd
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
    test_file = [data for data in current_data if data.endswith('test.parquet')][0]
        
    train_data = pd.read_parquet(path + f'/{train_file}')
    test_data = pd.read_parquet(path + f'/{test_file}')

    test_ids = test_data['id']

    train_data = train_data.drop(columns=['smpl', 'id'])
    test_data = test_data.drop(columns=['smpl', 'id'])

    task = Task('binary')

    roles = {
        'target': TARGET_NAME,
        #'drop': ['smpl','id']
        }
    
    automl = TabularAutoML(
        task = task, 
        timeout = TIMEOUT,
        cpu_limit = N_THREADS,
        reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},
    )

    out_of_fold_predictions = automl.fit_predict(train_data, roles = roles, verbose = 1)
    test_predictions = automl.predict(test_data)

    submission = pd.DataFrame({
        'id': test_ids.values,
        'target': test_predictions.data[:, 0]
    })

    prediction = submission[['id', 'target']].sort_values(by='id', ascending=True)

    return prediction

def model():
    data = 'data'
    folders = os.listdir(data)
   
    for fold in folders:
        data_path = data + f'/{fold}'

        prediction = fitting(path=data_path)
        if type(prediction) is not str:
            prediction.to_csv(f"predictions/{fold}.csv", index=False)
            print("Предсказание создано!")
        else:
            print("Невозможно создать предсказание!")

if __name__ == "__main__":
    model()
