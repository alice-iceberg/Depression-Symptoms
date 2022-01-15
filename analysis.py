import os

import pandas as pd

import tools


def combine_results(directory):
    frames = []
    filenames = os.listdir(directory)
    filenames = tools.remove_ds_store(filenames)

    for filename in filenames:
        suffix = f'_{filename.split("_")[0]}'
        df = pd.read_csv(f'{directory}/{filename}')
        pids = df['pid']
        df.drop(columns=['CV_TYPE', 'GT', 'pid'], axis=1, inplace=True)
        df = df.add_suffix(suffix)
        df['pid'] = pids
        frames.append(df)

    df_out = pd.concat(frames, axis=1)
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]
    df_out.to_csv(f'{directory}/combined_results.csv', index=False)


def combine_results_with_stats(filename):
    df = pd.read_csv(filename)
    df_phq_variance = pd.read_csv(f'{tools.TOOLS_PATH}/phq_stdev_per_pid.csv')
    df_num_samples = pd.read_csv(f'{tools.TOOLS_PATH}/samples_per_pid.csv')

    df_out = pd.merge(df, df_phq_variance, on='pid')
    df_out = pd.merge(df_out, df_num_samples, on='pid')
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]
    df_out.to_csv(f'{tools.RESULTS_PATH}/combined_results.csv', index=False)


def investigate_results(filename):
    df = pd.read_csv(filename)
    df_out = pd.DataFrame()
    df_out['zeros'] = df.isin([0]).sum(axis=1)
    df_out['pid'] = df['pid']
    df_out['phq'] = df['phq']
    df_out['samples'] = df['samples']
    df_out = df_out[df_out['phq'] >= 1]
    df_out = df_out[df_out['samples'] >= 50]

    df_out.sort_values(['phq', 'samples'], inplace=True, ascending=False)

    df_out.to_csv(f'{tools.TOOLS_PATH}/good_participants.csv', index=False)


def feature_importance_combine(directory):
    filenames = os.listdir(directory)
    filenames = tools.remove_ds_store(filenames)
    frames = []
    for filename in filenames:
        df = pd.read_csv(f'{directory}/{filename}')
        df.columns = ['feature', 'importance']
        df = df[df['feature'] != 'pid']
        frames.append(df)
    df_out = pd.concat(frames)
    symptom = directory.split('/')[-1]
    df_out.to_csv(f'importance/combined/{symptom}_combined.csv', index=False)


def feature_importance_investigate(directory):
    filenames = os.listdir(directory)
    filenames = tools.remove_ds_store(filenames)

    for filename in filenames:
        df = pd.read_csv(f'{directory}/{filename}')
        df = df.groupby(["feature"]).importance.sum().reset_index()
        df.sort_values(by=['importance'], inplace=True, ascending=False)
        df.to_csv(f'{directory}/{filename}', index=False)
