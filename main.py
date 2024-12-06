import pandas as pd
import os
import requests

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch

from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
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

    train_data = [data for data in current_data if data.endswith('train.parquet')][0]

    # Выделим тестовый датасет
    #test_data = [data for data in current_data if data.endswith('test.parquet')][0]
    # Откроем тренировочные данные
    train_data = pd.read_parquet(path + f'/{train_data}')
    # Откроем тестовые данные
    #test_data = pd.read_parquet(path + f'/{test_data}')


    train_data = train_data.sort_values(by='id')

    new_train_size = len(train_data) - int(len(train_data) * 0.2)
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

    out_of_fold_predictions = automl.fit_predict(new_train, roles = roles, verbose = 1)
    test_predictions = automl.predict(new_test)

    roc_auc_oot = roc_auc_score(new_test[TARGET_NAME].values, test_predictions.data[:, 0])

    end_time = time.time()  # Record the end time
    total_time = end_time - start_time
    minutes = int(total_time // 60)  # Calculate minutes
    seconds = int(total_time % 60)  # Calculate remaining seconds
    time_str = f"{minutes} minutes {seconds} seconds"  # Format the time string

    #model_pred = classificator.predict_proba(test_pool)[:, 1]
    # Объединим предсказание с метками
    #test_data['target'] = model_pred
    # Отсортируем предсказание
    #prediction = test_data[['id', 'target']].sort_values(by='id', ascending=True)
    # Вернем предсказание, как результат работы модели
    return {
        'time': time_str,
        'roc_auc_oot': roc_auc_oot,
        'train_size': new_train_size,
        'oot_size': len(train_data) - new_train_size,
        }

def model():

    stats = {
        "dataset": [],
        "train_size": [],
        "oot_size": [],
        "roc_auc_oot": [],
        "time": []
    }

    data = 'data'
    folders = os.listdir(data)
    
    for fold in folders:

        gc.collect()
        torch.cuda.empty_cache()

        data_path = data + f'/{fold}'
        
        prediction = fitting(path=data_path)


        stats['dataset'].append(str(fold))
        stats['train_size'].append(prediction['train_size'])
        stats['oot_size'].append(prediction['oot_size'])
        stats['roc_auc_oot'].append(prediction['roc_auc_oot'])
        stats['time'].append(prediction['time'])
        
        # if type(prediction) is not str:
        #     prediction.to_csv(f"predictions/{fold}.csv", index=False)
        #     print("Предсказание создано!")
        # else:
        #     print("Невозможно создать предсказание!")
    ans = pd.DataFrame(
        stats
    )

    ans.to_csv('./light_stats_20m_5cv.csv')

if __name__ == "__main__":
    model()
