import catboost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import tools


def preprocess_ema_variance(filename):
    df = pd.read_csv(filename)
    df.drop(columns=['var_sum', 'samples'], inplace=True, axis=1)
    df.set_index(['pid'], inplace=True)
    df = df.rank(1, ascending=True, method='first')
    df.to_csv(filename)


def variance_heatmap(filename):
    df = pd.read_csv(filename)
    columns = tools.SYMPTOM_ORIGINAL_COLUMN_LIST
    columns.insert(0, 'pid')
    df = df[columns]
    df.sort_values(by=['pid'], inplace=True)

    campaign = 'phase#1' if filename.__contains__('_4') else 'phase#2'
    group = 'depressed' if filename.__contains__('_dep_') else 'non-depressed'

    select_pid = []
    if campaign == 'phase#1' and group == 'depressed':
        select_pid = tools.SELECTED_PID_CMP4_DEP
    elif campaign == 'phase#2' and group == 'depressed':
        select_pid = tools.SELECTED_PID_CMP5_DEP
    elif campaign == 'phase#1' and group == 'non-depressed':
        select_pid = tools.SELECTED_PID_CMP4_NON_DEP
    elif campaign == 'phase#2' and group == 'non-depressed':
        select_pid = tools.SELECTED_PID_CMP5_NON_DEP

    df = df[df['pid'].isin(select_pid)]
    df.set_index(['pid'], inplace=True)

    title = f'Symptoms variance ranking ({campaign}: {group})'
    out_filename = f'symp_var_{campaign}_{group}.png'

    plt.figure(figsize=(8, 8), dpi=80)
    plt.title(title, fontdict={'fontsize': 15}, pad=8)
    sns.heatmap(df, cmap="Greens", vmin=1, vmax=9)
    plt.tight_layout()
    plt.yticks(rotation=0)
    plt.savefig(f'figures/symptoms/variance/{out_filename}')


def preprocess_ema_correlation(filename):
    df = pd.read_csv(filename)
    campaign = 'phase#1' if filename.__contains__('cmp4') else 'phase#2'
    group = 'depressed' if filename.__contains__('_dep_') else 'non-depressed'
    out_filename = f'symptoms_corr_phq_{campaign}_{group}.csv'

    drop_columns = list(df.columns)
    drop_columns = tools.subtract_lists(drop_columns, tools.SYMPTOM_ORIGINAL_COLUMN_LIST)
    drop_columns = tools.subtract_lists(drop_columns, ['pid', 'phq'])
    df.drop(columns=drop_columns, axis=1, inplace=True)

    select_pid = []
    if campaign == 'phase#1' and group == 'depressed':
        select_pid = tools.SELECTED_PID_CMP4_DEP
    elif campaign == 'phase#2' and group == 'depressed':
        select_pid = tools.SELECTED_PID_CMP5_DEP
    elif campaign == 'phase#1' and group == 'non-depressed':
        select_pid = tools.SELECTED_PID_CMP4_NON_DEP
    elif campaign == 'phase#2' and group == 'non-depressed':
        select_pid = tools.SELECTED_PID_CMP5_NON_DEP

    df = df[df['pid'].isin(select_pid)]

    # region finding correlations
    corr_df = pd.DataFrame()
    columns = list(df.columns)
    columns.remove('pid')

    for i in range(len(columns) - 1):
        corr_df = corr_df.append(df.groupby('pid')[columns].corr().stack()
                                 .loc[:, columns[i], columns[i + 1]:].reset_index())

    corr_df.columns = ['pid', 'symptom', 'v2', 'corr']
    corr_df = corr_df.set_index(['pid', 'symptom', 'v2']).sort_index()
    corr_df.reset_index(inplace=True)
    corr_df = corr_df.query('v2=="phq"')
    corr_df = corr_df.drop(columns=['v2'])

    corr_df_transpose = corr_df.pivot_table('corr', ['pid'], 'symptom')
    corr_df_transpose = corr_df_transpose.abs()
    corr_df_transpose = corr_df_transpose.fillna(0)
    corr_df_transpose.to_csv(f'symptoms/correlation/{out_filename}')
    # endregion


def correlation_heatmap(filename):
    df = pd.read_csv(filename)
    columns = tools.SYMPTOM_ORIGINAL_COLUMN_LIST
    columns.insert(0, 'pid')
    df = df[columns]
    df.sort_values(by=['pid'], inplace=True)
    df.set_index(['pid'], inplace=True)
    campaign = 'phase#1' if filename.__contains__('phase#1') else 'phase#2'
    group = 'non-depressed' if filename.__contains__('-depressed') else 'depressed'

    title = f'Symptoms vs PHQ score correlations ({campaign}: {group})'
    out_filename = f'symp_corr_{campaign}_{group}.png'

    plt.figure(figsize=(8, 8), dpi=80)
    plt.title(title, fontdict={'fontsize': 15}, pad=8)
    sns.heatmap(df, cmap="Greens", vmin=0, vmax=1)
    plt.tight_layout()
    plt.yticks(rotation=0)
    plt.savefig(f'figures/symptoms/correlation/{out_filename}')


def preprocess_ema_prediction(filename):
    df = pd.read_csv(filename)
    df['depressed'] = np.where(df['phq'] <= 14, 0, 1)
    campaign = 'phase#1' if filename.__contains__('cmp4') else 'phase#2'
    group = 'depressed' if filename.__contains__('_dep_') else 'non-depressed'
    out_filename = f'symptoms_pred_{campaign}_{group}.csv'

    drop_columns = list(df.columns)
    drop_columns = tools.subtract_lists(drop_columns, tools.SYMPTOM_ORIGINAL_COLUMN_LIST)
    drop_columns = tools.subtract_lists(drop_columns, ['pid', 'depressed'])
    df.drop(columns=drop_columns, axis=1, inplace=True)

    select_pid = []
    if campaign == 'phase#1' and group == 'depressed':
        select_pid = tools.SELECTED_PID_CMP4_DEP
    elif campaign == 'phase#2' and group == 'depressed':
        select_pid = tools.SELECTED_PID_CMP5_DEP
    elif campaign == 'phase#1' and group == 'non-depressed':
        select_pid = tools.SELECTED_PID_CMP4_NON_DEP
    elif campaign == 'phase#2' and group == 'non-depressed':
        select_pid = tools.SELECTED_PID_CMP5_NON_DEP

    df = df[df['pid'].isin(select_pid)]
    df.to_csv(f'symptoms/prediction/{out_filename}', index=False)


def train_val_test_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,
                                                      random_state=1)  # 0.25 x 0.8 = 0.2

    return X_train, y_train, X_val, y_val, X_test, y_test


def feature_importance_ema_prediction(filename):
    df = pd.read_csv(filename)
    extreme_pids_5_dep = [5021, 5039, 5057, 50157, 50198, 50200, 50281, 50523, 50546, 50559, 50588, 50715, 50741]
    extreme_pids_4_non_dep = [40171, 40174]
    extreme_pids_5_non_dep = [50106, 50251, 50577]

    campaign = 'phase#1' if filename.__contains__('phase#1') else 'phase#2'
    group = 'non-depressed' if filename.__contains__('non-depressed') else 'depressed'
    out_filename = f'symptoms_pred_imp_{campaign}_{group}.csv'

    extreme_pids = []
    if campaign == 'phase#1' and group == 'non-depressed':
        extreme_pids = extreme_pids_4_non_dep
    elif campaign == 'phase#2' and group == 'depressed':
        extreme_pids = extreme_pids_5_dep
    elif campaign == 'phase#2' and group == 'non-depressed':
        extreme_pids = extreme_pids_5_non_dep

    df = df[~df['pid'].isin(extreme_pids)]
    grouped = df.groupby('pid')
    frames = []

    for pid, group in tqdm(grouped):
        X = group[tools.SYMPTOM_ORIGINAL_COLUMN_LIST]
        y = group['depressed']

        X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

        # region machine learning
        cb_clf = catboost.CatBoostClassifier(random_seed=tools.RANDOM_SEED, depth=10, learning_rate=0.05,
                                             iterations=120,
                                             l2_leaf_reg=3)

        d_dev = catboost.Pool(
            data=X_train,
            label=y_train,
            feature_names=tools.SYMPTOM_ORIGINAL_COLUMN_LIST
        )

        d_val = catboost.Pool(
            data=X_val,
            label=y_val,
            feature_names=tools.SYMPTOM_ORIGINAL_COLUMN_LIST
        )

        cb_clf.fit(X=d_dev,
                   use_best_model=True,
                   eval_set=d_val,
                   verbose_eval=False,
                   early_stopping_rounds=35,
                   )

        # endregion

        # region feature importances
        feature_importances = pd.Series(cb_clf.feature_importances_, index=X.columns)
        df_feature_importances = pd.DataFrame(feature_importances).transpose()
        df_feature_importances = df_feature_importances[tools.SYMPTOM_ORIGINAL_COLUMN_LIST]
        df_feature_importances['pid'] = pid
        frames.append(df_feature_importances)
        # endregion
    for extreme_pid in extreme_pids:
        df_feature_importances = pd.DataFrame(np.nan, index=[extreme_pid], columns=tools.SYMPTOM_ORIGINAL_COLUMN_LIST)
        df_feature_importances.reset_index(inplace=True)
        df_feature_importances = df_feature_importances.rename(columns={'index': 'pid'})
        frames.append(df_feature_importances)

    df_out = pd.concat(frames)
    df_out.sort_values(by=['pid'], inplace=True)
    df_out.set_index(['pid'], inplace=True)
    df_out.to_csv(f'symptoms/prediction/{out_filename}')


def prediction_heatmap(filename):
    df = pd.read_csv(filename)
    columns = tools.SYMPTOM_ORIGINAL_COLUMN_LIST
    columns.insert(0, 'pid')
    df = df[columns]

    df.set_index(['pid'], inplace=True)
    df = df.rank(1, ascending=True, method='min')

    campaign = 'phase#1' if filename.__contains__('phase#1') else 'phase#2'
    group = 'non-depressed' if filename.__contains__('non-depressed') else 'depressed'

    title = f'Symptoms importance ranking ({campaign}: {group})'
    out_filename = f'symp_pred_{campaign}_{group}.png'

    plt.figure(figsize=(8, 8), dpi=80)
    plt.title(title, fontdict={'fontsize': 15}, pad=8)
    sns.heatmap(df, cmap="Greens", vmin=1, vmax=9)
    plt.tight_layout()
    plt.yticks(rotation=0)
    plt.savefig(f'figures/symptoms/prediction/{out_filename}')
