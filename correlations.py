import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import tools


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


def symptoms_vs_features_corr(filename, group):
    df = pd.read_csv(filename)

    if group == 'depressed':
        df = df[df['depressed'] == 1]
    elif group == 'non-depressed':
        df = df[df['depressed'] == 0]

    columns = tools.SYMPTOM_ORIGINAL_COLUMN_LIST
    M_features = df.columns.str.contains('#')
    feature_names = list(df.columns[M_features])
    columns.extend(feature_names)

    df = df[columns]
    df_corr = df.corr(method='spearman')
    df_corr.reset_index(inplace=True)
    df_corr.rename(columns={'index': 'feature'}, inplace=True)
    df_corr = df_corr[df_corr.feature.str.contains("#")]
    df_corr = df_corr[df_corr.columns.drop(list(df_corr.filter(regex='#')))]
    df_corr['abs_sum'] = df_corr[["lack_of_interest",
                                  "depressed_feeling",
                                  "sleep_trouble",
                                  "fatigue",
                                  "poor_appetite",
                                  "negative_self_image",
                                  "difficulty_focusing",
                                  "bad_physchomotor_activity",
                                  "suicide_thoughts"]].abs().sum(axis=1)
    df_corr.sort_values(by=['abs_sum'], ascending=False, inplace=True)
    df_corr = df_corr.head(30)
    df_corr = df_corr.drop(columns=['abs_sum'])

    df_corr.to_csv(f'correlation/WIN4/symptoms/sym_feat_WIN4_{group}.csv', index=False)

    title = f'Features vs symptoms correlations ({group})'
    out_filename = f'correlation/WIN4/symptoms/symp_feat_{group}.png'
    df_corr.set_index(['feature'], inplace=True)
    plt.figure(figsize=(8, 8), dpi=80)
    plt.title(title, fontdict={'fontsize': 15}, pad=8)
    sns.heatmap(df_corr, cmap="bwr", vmin=-0.3, vmax=0.3)
    plt.tight_layout()
    plt.yticks(rotation=0)
    plt.savefig(out_filename)
