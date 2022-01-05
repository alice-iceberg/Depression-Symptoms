import preprocess
from tools import FEATURES_PATH


def main():
    preprocess.get_missing_data_per_feature(preprocess.process_missing_data(preprocess.filter_dataframe(FEATURES_PATH)))


if __name__ == '__main__':
    main()
