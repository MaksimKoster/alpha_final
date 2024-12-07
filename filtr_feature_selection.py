import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

import joblib

def calc_metric(feature_name, y_true, y_pred):
    return (feature_name, abs(2 * roc_auc_score(y_true, y_pred) - 1))
    

def filtr_feat_selection(train_df, features, target_col='target', n_jobs=joblib.cpu_count()-1):
    train_df['random_feat'] = np.random.random(train_df.shape[0])

    metric_cutoff = abs(2 * roc_auc_score(train_df[target_col], train_df['random_feat']) - 1)

    delayed_funcs = [joblib.delayed(calc_metric)(i, train_df[target_col], train_df[i]) for i in features]
    parallel_pool = joblib.Parallel(n_jobs=n_jobs)

    res = parallel_pool(delayed_funcs)

    return [feat_name for feat_name, roc_auc in res if roc_auc > metric_cutoff]



def light_rf_feature_selection(train_df, features, target_col='target', n_jobs=joblib.cpu_count()-1):
    train_df['random_feat'] = np.random.random(train_df.shape[0])

    features = np.array(features + ['random_feat'])
    
    rf = RandomForestClassifier(
        n_estimators=500, 
        max_depth=7, 
        n_jobs=n_jobs, 
        min_samples_split=int(train_df.shape[0]*0.05),
        min_samples_leaf=int(train_df.shape[0]*0.025),
        max_features='log2',
        max_samples=0.5
    )

    rf.fit(train_df[features], train_df[target_col])

    feature_imp = dict(zip(features, rf.feature_importances_))

    cutoff = feature_imp.get('random_feat')

    return features[np.array(list(feature_imp.values())) > cutoff]