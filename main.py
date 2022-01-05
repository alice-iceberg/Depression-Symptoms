import pandas as pd

from tqdm import tqdm
import ml
import preprocess
import tools


def main():
    df = pd.read_csv(tools.PREPROCESSED_FEATURES_PATH)

    for gt in tqdm(tools.SYMPTOM_BIN_COLUMN_LIST):
        df_out = ml.run_classification(df, gt)
        df_out.to_csv(f'results/{gt}.csv', index=True)


if __name__ == '__main__':
    main()
