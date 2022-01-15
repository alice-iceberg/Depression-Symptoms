import warnings
from functools import reduce

import catboost
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt

import preprocess
import tools
from tools import specificity_score


def perf_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    metrics = []
    y_pred_cls = np.rint(y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1 = f1_score(y_true, y_pred_cls)
        prec_score = precision_score(y_true, y_pred_cls)
        rec_score = recall_score(y_true, y_pred_cls)
        #spec_score = specificity_score(y_true, y_pred_cls)

    metrics.append(dict(F1=f1, PRECISION=prec_score, RECALL=rec_score))
    return reduce(lambda a, b: dict(a, **b), metrics)


def LOSOCatBoost(X, y, feature_names, pids, gt):
    splitter = GroupKFold(n_splits=len(np.unique(pids)))
    results = []
    for i, (train_index, test_index) in enumerate(splitter.split(X, y, groups=pids)):
        results.append(run_trial(X, y, train_index, test_index, feature_names, gt))
    results = pd.DataFrame(results)
    results.insert(0, 'CV_TYPE', 'LOSO', allow_duplicates=True)
    results.insert(0, 'GT', gt, allow_duplicates=True)

    return results


def run_trial(X, y, train_index, test_index, feature_names, gt):
    pids = X['pid']
    pid = X['pid'].iloc[test_index].unique()[0]
    X = X[feature_names]

    cb_clf = catboost.CatBoostClassifier(random_seed=tools.RANDOM_SEED, depth=10, learning_rate=0.05, iterations=120,
                                         l2_leaf_reg=3)
    dev_index, val_index = group_inner_split(X.iloc[train_index][feature_names], y.iloc[train_index],
                                             pids.iloc[train_index])

    d_dev = catboost.Pool(
        data=X.iloc[train_index].iloc[dev_index][feature_names],
        label=y.iloc[train_index].iloc[dev_index],
        feature_names=feature_names
    )

    d_val = catboost.Pool(
        data=X.iloc[train_index].iloc[val_index][feature_names],
        label=y.iloc[train_index].iloc[val_index],
        feature_names=feature_names
    )

    cb_clf.fit(X=d_dev,
               use_best_model=True,
               eval_set=d_val,
               verbose_eval=False,
               early_stopping_rounds=35,
               )

    prob = cb_clf.predict_proba(X.iloc[test_index][feature_names])[:, 1]
    metrics = perf_metrics(y.iloc[test_index], prob)
    res = metrics
    res.update({'pid': pid})

    feat_importances = pd.Series(cb_clf.feature_importances_, index=X.columns)
    save_feature_importance(pid, feat_importances, gt)

    return res


def group_inner_split(X_train, y_train, pids):
    inner_splitter = GroupKFold(n_splits=5)
    for dev_index, val_index in inner_splitter.split(X_train, y_train, groups=pids):
        return dev_index, val_index


def run_classification(args):
    df = args[0]
    gt = args[1]
    M_features = df.columns.str.contains('#')
    feature_names = list(df.columns[M_features])
    feature_names.append('pid')
    X = df[feature_names]

    # X = preprocess.normalize_dataframe(X, feature_names)
    # X = preprocess.binnarize_dataframe(X, feature_names)
    y = df[gt]
    pids = df['pid']
    feature_names.remove('pid')

    return [LOSOCatBoost(X, y, feature_names, pids, gt), gt]


def save_feature_importance(pid, feature_importances, gt):
    feature_importances.sort_values(inplace=True, ascending=False)
    feature_importances = feature_importances.head(20)
    feature_importances['pid'] = pid
    feature_importances.to_csv(f'importance/{gt}_{pid}.csv')
