import pandas as pd

import tools


def filter_dataframe(filename):
    df = pd.read_csv(filename)
    print(f'Original Dataframe: {df.shape}')

    M_drop_participant = df['pid'].isin(tools.DROP_PARTICIPANT_LIST)
    df = df[~M_drop_participant]
    print(f'Dataframe after dropping participants: {df.shape}')

    M_ema_filter = (df['phq'] == 9) & (df['duration'] <= 9)
    df = df[~M_ema_filter]
    print(f'Dataframe after EMA filter: {df.shape}')

    M_missing_unlockdata = df['missing_unlockState_1days'] == 1
    df = df[~M_missing_unlockdata]
    print(f'Dataframe after missing unlock data: {df.shape}')

    return df


def get_missing_data_per_feature(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_missing': percent_missing})
    missing_value_df.sort_values(by=['percent_missing'], ascending=False, inplace=True)

    tools.create_dir_if_not_exists(tools.TOOLS_PATH)
    missing_value_df.to_csv(f'{tools.TOOLS_PATH}/missing4hr.csv', index=False)


def process_missing_data(df):
    df.drop(columns=tools.DROP_FEATURES_LIST, axis=1, inplace=True)  # drop features
    df.update(df[tools.IMPUTE_ZERO_FEATURES_LIST].fillna(0))  # fillna with 0 values

    df = df[~(df.isnull().sum(axis=1) > 54)]  # drop rows with more than 54 (54 is 25% of 216 features) missing values
    df = tools.fillna_mean_bygroup(df, list(df.columns), 'pid')  # fillna by group (pid) mean
    df = df[~(df.isnull().sum(axis=1) >= 1)]  # drop rows with any missing values

    print(f'Dataframe after missing data preprocessing: {df.shape}')
    df.to_csv(tools.PREPROCESSED_FEATURES_PATH, index=False)

    return df
