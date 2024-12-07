from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import os
import shutil
import gc

def fitting_autogluon(path):
    try:
        current_data = os.listdir(path)
    except Exception:
        return "Не папка"
    else:
        train_file = [data for data in current_data if data.endswith('train.parquet')][0]
        test_file = [data for data in current_data if data.endswith('test.parquet')][0]
        
        train_data = pd.read_parquet(path + f'/{train_file}')
        test_data = pd.read_parquet(path + f'/{test_file}')
        
        test_ids = test_data['id']
        
        train_data = train_data.drop(columns=['smple', 'id'])
        
        test_data = test_data.drop(columns=['smpl', 'id'])
        
        label = 'target'

        num_gpus = 1
        num_cpus = 16

        os.environ['OMP_NUM_THREADS'] = str(num_cpus)
        os.environ['OPENBLAS_NUM_THREADS'] = str(num_cpus)
        os.environ['MKL_NUM_THREADS'] = str(num_cpus)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_cpus)
        os.environ['NUMEXPR_NUM_THREADS'] = str(num_cpus)
        
        predictor = TabularPredictor(label=label, path=f'ag_models', eval_metric='roc_auc')
        
        predictor.fit(train_data=train_data, time_limit=1500, num_gpus=1, num_cpus=16)
        
        predictions_proba = predictor.predict_proba(test_data)
        
        positive_class = predictor.classes_[1]
        predictions = predictions_proba[positive_class]
        
        prediction_df = pd.DataFrame({'id': test_ids, 'target': predictions})
        
        prediction_df = prediction_df.sort_values(by='id', ascending=True)
        
        return prediction_df

def model_autogluon():
    data_dir = 'data'
    folders = os.listdir(data_dir)
    
    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    
    for fold in folders:
        data_path = os.path.join(data_dir, fold)
        prediction = fitting_autogluon(data_path)
        
        if isinstance(prediction, pd.DataFrame):
            prediction.to_csv(f"predictions/{fold}.csv", index=False)
            print(f"Предсказание для {fold} создано!")

            gc.collect()

            for item in os.listdir('ag_models'):
                item_path = os.path.join('ag_models', item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                        print(f'Deleted file: {item_path}')
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        print(f'Deleted directory: {item_path}')
                except Exception as e:
                    print(f'Error deleting {item_path}: {e}')

        else:
            print(f"Невозможно создать предсказание для {fold}")

if __name__ == "__main__":
    model_autogluon()