import concurrent.futures
from datetime import datetime

import pandas as pd
from tqdm import tqdm

import correlations
import ml
import tools


def machine_learning():
    df_original = pd.read_csv(tools.PREPROCESSED_FEATURES_PATH)
    remove_pids_5_dep = [50692, 50573, 50704, 50576, 50198, 50200, 50715, 50588, 5021, 50472, 5039, 50480, 50737, 50741,
                         50358, 50360, 5049, 50235, 5057, 50380, 5073, 50523, 50281, 50157, 50546, 50292, 50559]
    remove_pids_4_nondep = [40163, 40164, 40171, 40141, 40114]
    remove_pids_5_nondep = [50577, 50194, 50583, 50463, 50344, 50221, 50222, 50224, 50742, 50103, 50106, 50624, 5062,
                            50631, 50119, 5065, 50251, 50125, 50134, 5079, 5081, 50652, 5084, 50408, 50409, 50284,
                            50687]
    for depressed in ['depressed', 'non-depressed']:
        for campaign in [tools.CAMPAIGN_4, tools.CAMPAIGN_5]:
            print(f'{datetime.now()}: {campaign} {depressed}')

            if depressed == 'depressed' and campaign == tools.CAMPAIGN_4:
                pid_list = tools.SELECTED_PID_CMP4_DEP
                continue
            elif depressed == 'non-depressed' and campaign == tools.CAMPAIGN_4:
                pid_list = tools.subtract_lists(tools.SELECTED_PID_CMP4_NON_DEP, remove_pids_4_nondep)
                continue
            elif depressed == 'depressed' and campaign == tools.CAMPAIGN_5:
                pid_list = tools.subtract_lists(tools.SELECTED_PID_CMP5_DEP, remove_pids_5_dep)
                continue
            elif depressed == 'non-depressed' and campaign == tools.CAMPAIGN_5:
                pid_list = tools.subtract_lists(tools.SELECTED_PID_CMP5_NON_DEP, remove_pids_5_nondep)
            elif depressed == 'all' and campaign == tools.CAMPAIGN_4:
                pid_list = tools.SELECTED_PID_CMP4_DEP
                pid_list.extend(tools.SELECTED_PID_CMP4_NON_DEP)
            elif depressed == 'all' and campaign == tools.CAMPAIGN_5:
                pid_list = tools.SELECTED_PID_CMP5_DEP
                pid_list.extend(tools.SELECTED_PID_CMP5_NON_DEP)
            else:
                pid_list = []

            df = df_original[df_original['pid'].isin(pid_list)]
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = [executor.submit(ml.run_personalized_classification, [df, symptom]) for symptom in
                           tqdm(tools.SYMPTOM_BIN_COLUMN_LIST)]

                for f in concurrent.futures.as_completed(results):
                    args = f.result()
                    df_out = args[0]
                    symptom = args[1]
                    df_out.to_csv(
                        f'results/cmp45_Jan/personalized/cmp{campaign}/{depressed}/{symptom}_30percent_pers_{depressed}.csv',
                        index=False)


def describe_combined_file(filename):
    group = filename.split('/')[-2]
    campaign = filename.split('/')[-3]
    cv_method = filename.split('/')[2]

    out_filename = f'stats/cmp45_Jan/{cv_method}/{campaign}_{group}_{cv_method}.csv'

    df = pd.read_csv(filename)
    df_desc = df.describe()
    df_desc = df_desc.round(2)

    df_desc.to_csv(out_filename)


def main():
    folders = ['results/cmp45_Jan/personalized/cmp4/depressed/combined_results.csv',
               'results/cmp45_Jan/personalized/cmp4/non-depressed/combined_results.csv',
               'results/cmp45_Jan/personalized/cmp5/depressed/combined_results.csv',
               'results/cmp45_Jan/personalized/cmp5/non-depressed/combined_results.csv',
               'results/cmp45_Jan/stratified/cmp4/all/combined_results.csv',
               'results/cmp45_Jan/stratified/cmp4/depressed/combined_results.csv',
               'results/cmp45_Jan/stratified/cmp4/non-depressed/combined_results.csv',
               'results/cmp45_Jan/stratified/cmp5/all/combined_results.csv',
               'results/cmp45_Jan/stratified/cmp5/depressed/combined_results.csv',
               'results/cmp45_Jan/stratified/cmp5/non-depressed/combined_results.csv'
               ]
    for folder in folders:
        describe_combined_file(folder)


if __name__ == '__main__':
    correlations.phq_score_vs_features_corr(tools.PREPROCESSED_FEATURES_PATH, 'non-depressed')