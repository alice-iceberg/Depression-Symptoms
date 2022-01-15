import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def preprocess_ema_variance(filename):
    df = pd.read_csv(filename)
    df.drop(columns=['var_sum', 'samples'], inplace=True, axis=1)
    df.set_index(['pid'], inplace=True)
    df = df.rank(1, ascending=True, method='first')
    df.to_csv(filename)


def variance_heatmap(filename):
    df = pd.read_csv(filename)
    df.set_index(['pid'], inplace=True)
    campaign = 'phase#1' if filename.__contains__('_4') else 'phase#2'
    group = 'depressed' if filename.__contains__('_dep_') else 'non-depressed'

    title = f'Symptoms variance ranking ({campaign}: {group})'
    out_filename = f'symp_var_{campaign}_{group}.png'

    plt.figure(figsize=(8, 8), dpi=80)
    plt.title(title, fontdict={'fontsize': 15}, pad=8)
    sns.heatmap(df, cmap="Greens")
    plt.tight_layout()
    plt.savefig(f'figures/symptoms/variance/{out_filename}')
