import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
    print(campaign, group)
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

