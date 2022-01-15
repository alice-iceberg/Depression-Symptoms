import concurrent.futures
import os
from datetime import datetime

import pandas as pd
from tqdm import tqdm

import analysis
import ml
import symptoms_importance
import tools


def machine_learning():
    df_original = pd.read_csv(tools.PREPROCESSED_FEATURES_PATH)

    for depressed in ['depressed', 'non-depressed']:
        for campaign in [tools.CAMPAIGN_4, tools.CAMPAIGN_5]:
            print(f'{datetime.now()}: {campaign} {depressed}')

            if depressed == 'depressed' and campaign == tools.CAMPAIGN_4:
                pid_list = tools.SELECTED_PID_CMP4_DEP
            elif depressed == 'non-depressed' and campaign == tools.CAMPAIGN_4:
                pid_list = tools.SELECTED_PID_CMP4_NON_DEP
            elif depressed == 'depressed' and campaign == tools.CAMPAIGN_5:
                pid_list = tools.SELECTED_PID_CMP5_DEP
            elif depressed == 'non-depressed' and campaign == tools.CAMPAIGN_5:
                pid_list = tools.SELECTED_PID_CMP5_NON_DEP
            else:
                pid_list = []

            df = df_original[df_original['pid'].isin(pid_list)]
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = [executor.submit(ml.run_classification, [df, symptom]) for symptom in
                           tqdm(tools.SYMPTOM_BIN_COLUMN_LIST)]

                for f in concurrent.futures.as_completed(results):
                    args = f.result()
                    df_out = args[0]
                    symptom = args[1]
                    df_out.to_csv(f'results/cmp45_Jan/cmp{campaign}/{depressed}/{symptom}_30percent_{depressed}.csv',
                                  index=False)


def main():
    filename = 'results/cmp45_Jan/cmp4/all/combined_results.csv'
    out_filename = 'stats/cmp45_Jan/cmp4_all_30perc.csv'
    df = pd.read_csv(filename)
    df_desc = df.describe()
    df_desc = df_desc.round(2)

    df_desc.to_csv(out_filename)


if __name__ == '__main__':
    symptoms_importance.variance_heatmap('symptoms/variance/combined_stdev_nondep_4.csv')
