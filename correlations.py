import pandas as pd
from scipy.stats import pearsonr


def phq_score_vs_features_corr(filename, group):
    df_original = pd.read_csv(filename)

    if group == 'depressed':
        df_original = df_original[df_original['depressed'] == 1]
    elif group == 'non-depressed':
        df_original = df_original[df_original['depressed'] == 0]

    M_features = df_original.columns.str.contains('#')
    feature_names = list(df_original.columns[M_features])
    df_features = df_original[feature_names]

    series_corr = df_features.corrwith(df_original['phq'])
    df_corr = pd.DataFrame(series_corr)
    df_corr.reset_index(inplace=True)
    df_corr.columns = ['feature', 'corr']
    p_values = calculate_pvalues(df_features, df_original['phq'])

    df_corr['p_value'] = p_values
    df_corr['corr_abs'] = abs(df_corr['corr'])

    df_corr.sort_values(by=['corr_abs'], ascending=False, inplace=True)
    df_corr.drop(columns=['corr_abs'], inplace=True, axis=1)
    df_corr.to_csv(f'correlation/WIN4/phq_feat_WIN4_{group}.csv', index=False)


def calculate_pvalues(df_features, phq):
    pvalues = []
    for feature in df_features.columns:
        pvalues.append(round(pearsonr(df_features[feature], phq)[1], 4))
    return pvalues
