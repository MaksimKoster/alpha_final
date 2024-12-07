import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import VarianceThreshold

import time

import joblib

def calc_metric(feature_name, y_true, y_pred):
    return (feature_name, abs(2 * roc_auc_score(y_true.reshape(-1, 1), y_pred.reshape(-1, 1)) - 1))


def create_new_features(df):
    df['sum_by_row'] = df.drop(columns=['target','smpl','id']).sum(axis=1)
    df['mean_by_row'] = df.drop(columns=['target','smpl','id', 'sum_by_row']).mean(axis=1)
    df['variance_by_row'] = df.drop(columns=['target','smpl','id', 'mean_by_row', 'sum_by_row']).var(axis=1)
    df['std_by_row'] = df.drop(columns=['target','smpl','id', 'mean_by_row', 'sum_by_row', 'variance_by_row']).std(axis=1)
    df['fskew'] = df.drop(columns=['smpl','id', 'mean_by_row', 'sum_by_row', 'variance_by_row', 'std_by_row']).skew(axis=1)
    df['fkurtosis'] = df.drop(columns=['smpl','id', 'mean_by_row', 'sum_by_row', 'variance_by_row', 'fskew', 'std_by_row']).kurtosis(axis=1)
    df['id_sin'] = np.sin(df['id'])
    df['id_cos'] = np.cos(df['id'])

    return df
    

def filtr_feat_selection(train_df, features, target_col='target', n_jobs=joblib.cpu_count()-1, random_state=42):
    temp_df = train_df.copy()

    np.random.seed(random_state)
    
    temp_df['random_feat'] = np.random.random(temp_df.shape[0])

    _, metric_cutoff = calc_metric('', temp_df[target_col].values, temp_df['random_feat'].values)

    delayed_funcs = [joblib.delayed(calc_metric)(i, temp_df[target_col].values, temp_df[i].values) for i in features]
    parallel_pool = joblib.Parallel(n_jobs=n_jobs)

    res = parallel_pool(delayed_funcs)

    return [feat_name for feat_name, roc_auc in res if roc_auc > metric_cutoff]


def light_rf_feature_selection(train_df, features, target_col='target', n_jobs=joblib.cpu_count()-1, random_state=42):
    np.random.seed(random_state)
    
    temp_df = train_df.copy()
    temp_df['random_feat'] = np.random.random(temp_df.shape[0])

    features = np.array(features + ['random_feat'])
    
    rf = RandomForestClassifier(
        n_estimators=250, 
        max_depth=16, 
        n_jobs=n_jobs, 
        min_samples_split=int(temp_df.shape[0]*0.05),
        min_samples_leaf=int(temp_df.shape[0]*0.025),
        max_features='sqrt',
        max_samples=0.5,
        random_state=random_state
    )

    rf.fit(temp_df[features], temp_df[target_col])

    feature_imp = dict(zip(features, rf.feature_importances_))

    cutoff = feature_imp.get('random_feat')

    return list(features[np.array(list(feature_imp.values())) > cutoff])


def light_lgbm_feature_selection(train_df, features, target_col='target', n_jobs=joblib.cpu_count()-1, random_state=42):
    np.random.seed(random_state)
    
    temp_df = train_df.copy()
    temp_df['random_feat'] = np.random.random(temp_df.shape[0])

    features = np.array(features + ['random_feat'])
    
    clf = LGBMClassifier(
        boosting_type='rf',
        n_estimators=146,  # Number of trees
        max_depth=16,  # Maximum depth of trees
        min_child_samples=211,  # Minimum number of samples per leaf
        min_child_weight=649,  # Minimum sum of weights of all observations required in a child
        n_jobs=16,
        random_state=random_state,
        bagging_freq=1,  # Frequency for bagging
        bagging_fraction=0.9,  # Fraction of data to be used for bagging
        feature_fraction=0.9,  # Fraction of features to be used for training
        subsample=None,
        colsample_bytree=None,
        subsample_freq=None,
        verbose=-1,
        # device='gpu',
        gpu_device_id=0
    )
    clf.fit(temp_df[features], temp_df[target_col])

    feature_imp = dict(zip(features, clf.feature_importances_))

    cutoff = feature_imp.get('random_feat')

    return list(features[np.array(list(feature_imp.values())) > cutoff])


def feature_selection_wrapper(df, rf_cls_type=True):
    selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
    selector.fit_transform(df.drop(columns=['id', 'smpl', 'target']))

    var_short_list = list(selector.feature_names_in_) 
    
    random_state = int(time.time())

    if rf_cls_type:
        short_list = light_rf_feature_selection(df, var_short_list, 'target', random_state=random_state)
    else:
        short_list = light_lgbm_feature_selection(df, var_short_list, 'target', random_state=random_state)
        
                  
    return short_list


def df_prep(df):
    df = create_new_features(df)

    short_list = feature_selection_wrapper(df)

    return df[short_list + ['id', 'smpl', 'target']]