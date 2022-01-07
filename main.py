import os

import pandas as pd
from tqdm import tqdm

import analysis
import ml
import tools


def machine_learning():
    df = pd.read_csv(tools.PREPROCESSED_FEATURES_PATH)

    df = df[df['pid'].isin(tools.SELECTED_PID_BY_PHQ_SAMPLES)]
    print(df.shape)
    for symptom in tqdm(tools.SYMPTOM_BIN_COLUMN_LIST):
        df_out = ml.run_classification(df, symptom)
        df_out.to_csv(f'results/{symptom}_sel.csv', index=False)


def main():
    # analysis.combine_results('results/selected')

    filename = 'results/selected/combined_results_sel.csv'
    df = pd.read_csv(filename)
    result_stats = df.describe()
    result_stats.drop(['pid'], axis=1, inplace=True)
    result_stats.to_csv('stats/sel_stats.csv')


if __name__ == '__main__':
    analysis.feature_importance_investigate(f'importance/combined')