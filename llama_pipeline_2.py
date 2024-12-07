import os
import gc
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
)

from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML
from lightautoml.tasks import Task

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set seeds and constants
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
N_THREADS = -1 #16
N_FOLDS = 10
TIMEOUT = 1800  # 30 minutes as an example
DATA_DIR = "/home/rbparchiev/alpha_hackathon/alpha_step_2/data/data_fs_rf/"
OUTPUT_DIR = os.path.join(DATA_DIR, "lightautoml_pipeline_results_pandas")
SUBMISSION_DIR = os.path.join(OUTPUT_DIR, "submissions")
os.makedirs(SUBMISSION_DIR, exist_ok=True)

datasets = [
    "fl_credit_card_tendency",
    "invest_prop_4",
    "outflow_12",
    "pd_fl",
    "pd_ul_9",
    "ul_leasing_outflow"
]

torch.set_num_threads(N_THREADS)


def load_data(dataset_name):
    train_path = os.path.join(DATA_DIR, f"{dataset_name}_train_selected.parquet")
    test_path = os.path.join(DATA_DIR, f"{dataset_name}_test_selected.parquet")

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    return df_train, df_test


def find_best_threshold(y_true, y_proba):
    """
    Find threshold that maximizes F1 score.
    Search over thresholds in [0.0, 1.0] with step 0.01.
    """
    best_thr = 0.5
    best_f1 = -1
    thresholds = np.linspace(0.0, 1.0, 101)
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


def print_metrics(y_true, y_proba, threshold):
    """
    Print multiple metrics:
    - AUC
    - Average Precision
    - Precision, Recall, F1 at given threshold
    """
    auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    print(f"AUC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print(f"At threshold {threshold:.2f}: F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")


def make_submission(test_ids, test_proba, dataset_name):
    submission = pd.DataFrame({
        "id": test_ids,
        "prediction": test_proba
    })
    sub_path = os.path.join(SUBMISSION_DIR, f"{dataset_name}_submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to: {sub_path}")


def process_dataset(dataset_name):
    print(f"Processing dataset: {dataset_name}")
    start_time = time.time()

    df_train, df_test = load_data(dataset_name)

    test_ids = df_test['id'].values
    train_data = df_train.drop(columns=['id', 'smpl'])
    test_data = df_test.drop(columns=['id', 'smpl'])

    y_train = train_data['target'].values

    task = Task('binary')
    roles = {
        'target': 'target'
    }

    automl = TabularUtilizedAutoML(
        task=task,
        timeout=TIMEOUT,
        cpu_limit=N_THREADS,
        gpu_ids='0',
        reader_params={'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': SEED}
    )

    oof_preds = automl.fit_predict(train_data, roles=roles, verbose=1)
    y_oof_proba = oof_preds.data[:, 0]

    best_thr, best_f1 = find_best_threshold(y_train, y_oof_proba)

    print("OOF Metrics:")
    print_metrics(y_train, y_oof_proba, best_thr)

    test_preds = automl.predict(test_data)
    y_test_proba = test_preds.data[:, 0]

    print("Test predictions stats:")
    print(f"Mean: {y_test_proba.mean():.4f}, Std: {y_test_proba.std():.4f}")

    make_submission(test_ids, y_test_proba, dataset_name)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Dataset {dataset_name} processing time: {elapsed:.2f} seconds")

    del df_train, df_test, train_data, test_data, oof_preds, test_preds, automl
    gc.collect()


if __name__ == "__main__":
    total_start = time.time()
    for dataset_name in datasets:
        process_dataset(dataset_name)
    total_end = time.time()
    total_time_spent = total_end - total_start
    print(f"All done! Total time spent for all datasets: {total_time_spent:.2f} seconds")
