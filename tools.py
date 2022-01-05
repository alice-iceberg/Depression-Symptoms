import os

from sklearn.metrics import confusion_matrix

import tools

RANDOM_SEED = 44
FEATURES_PATH = '/Users/aliceberg/Documents/Paper2022/data/FS_all_sensors_campaign_4_5-WIN_4.csv'
PREPROCESSED_FEATURES_PATH = 'data/FS_all_sensors_campaign_4_5-WIN_4_preprocessed.csv'
TOOLS_PATH = 'tools'
DROP_FEATURES_LIST = ["mediaStorage#4HR#MUSIC_VAR",
                      "mediaStorage#4HR#VIDEO_VAR",
                      "mediaStorage#4HR#IMAGE_VAR",
                      "call#4HR#IN_dur_VAR",
                      "call#4HR#OUT_dur_VAR",
                      "call#4HR#IN_dur_ASC",
                      "call#4HR#IN_dur_KUR",
                      "call#4HR#IN_dur_SKW",
                      "call#4HR#OUT_dur_ASC",
                      "call#4HR#OUT_dur_KUR",
                      "call#4HR#OUT_dur_SKW",
                      "call#4HR#ratio_in2out",
                      "sms#4HR#num_char_SKW",
                      "sms#4HR#num_char_VAR",
                      "sms#4HR#num_char_KUR",
                      "sms#4HR#num_char_ASC",
                      "location#4HR#norm_entropy",
                      "sound#4HR#PITCH_VAR",
                      "sound#4HR#PITCH_KUR",
                      "sound#4HR#PITCH_SKW",
                      "sound#4HR#PITCH_ASC",
                      "wifi#4HR#num_APs_SKW",
                      "wifi#4HR#num_APs_ASC",
                      "wifi#4HR#num_APs_KUR",
                      "wifi#4HR#num_APs_VAR",
                      "unlock#4HR#VAR",
                      "typing#4HR#typingDur_VAR",
                      "mediaStorage#4HR#MUSIC_SKW",
                      "mediaStorage#4HR#VIDEO_SKW",
                      "mediaStorage#4HR#IMAGE_SKW",
                      "mediaStorage#4HR#VIDEO_KUR",
                      "mediaStorage#4HR#IMAGE_KUR",
                      "mediaStorage#4HR#MUSIC_KUR",
                      "mediaStorage#4HR#IMAGE_ASC",
                      "mediaStorage#4HR#MUSIC_ASC",
                      "mediaStorage#4HR#VIDEO_ASC",
                      "mediaStorage#4HR#IMAGE_MIN",
                      "mediaStorage#4HR#VIDEO_MIN",
                      "mediaStorage#4HR#MUSIC_MIN",
                      "mediaStorage#4HR#IMAGE_MAX",
                      "mediaStorage#4HR#VIDEO_MAX",
                      "mediaStorage#4HR#MUSIC_MAX",
                      "duration"
                      ]
IMPUTE_ZERO_FEATURES_LIST = ["activity#4HR#RUNNING#dur",
                             "activity#4HR#RUNNING#freq",
                             "activity#4HR#ON_BICYCLE#freq",
                             "activity#4HR#ON_BICYCLE#dur",
                             "activity#4HR#IN_VEHICLE#freq",
                             "activity#4HR#IN_VEHICLE#dur",
                             "activity#4HR#WALKING#dur",
                             "activity#4HR#WALKING#freq",
                             "activity#4HR#STILL#dur",
                             "activity#4HR#STILL#freq",
                             "call#4HR#IN_dur_MAX",
                             "call#4HR#IN_dur_MIN",
                             "call#4HR#IN_dur_AVG",
                             "call#4HR#IN_dur_MED",
                             "call#4HR#OUT_dur_AVG",
                             "call#4HR#OUT_dur_MAX",
                             "call#4HR#OUT_dur_MED",
                             "call#4HR#OUT_dur_MIN",
                             "sms#4HR#num_char_MED",
                             "sms#4HR#num_char_MAX",
                             "sms#4HR#num_char_MIN",
                             "sms#4HR#num_char_AVG",
                             "unlock#4HR#MIN",
                             "unlock#4HR#MED",
                             "unlock#4HR#MAX",
                             "unlock#4HR#AVG",
                             "typing#4HR#typingDur_MAX",
                             "typing#4HR#typingDur_MIN",
                             "typing#4HR#typingDur_MED",
                             "typing#4HR#typingDur_AVG",
                             "typing#4HR#uniqueApps",
                             "keystrk#4HR#num_sessions",
                             "notif#4HR#ARRIVED_freq",
                             "notif#4HR#CLICKED_freq",
                             "notif#4HR#DECISION_TIME_freq",
                             "typing#4HR#freq",
                             ]

SYMPTOM_ORIGINAL_COLUMN_LIST = ["lack_of_interest",
                                "depressed_feeling",
                                "sleep_trouble",
                                "fatigue",
                                "poor_appetite",
                                "negative_self_image",
                                "difficulty_focusing",
                                "bad_physchomotor_activity",
                                "suicide_thoughts"]

SYMPTOM_BIN_COLUMN_LIST = ["lack_of_interest_bin",
                           "depressed_feeling_bin",
                           "sleep_trouble_bin",
                           "fatigue_bin",
                           "poor_appetite_bin",
                           "negative_self_image_bin",
                           "difficulty_focusing_bin",
                           "bad_physchomotor_activity_bin",
                           "suicide_thoughts_bin"
                           ]

DROP_PARTICIPANT_LIST = [4082, 4084, 4096,  # Android version problem
                         5072, 50196, 50189, 50417, 50370, 50215,  # 0 variation in total phq score
                         50179, 50230, 50538, 50628, 50630,  # have 1 EMA only
                         50702, 50710, 50769, 50514, 50286, 50477, 50771, 50765, 50184, 50365, 50389, 50759, 50559,
                         50513, 50343, 50692, 50742, 50741, 50738, 50691, 50700, 50577, 50733]  # have less than 10 EMAs


def create_dir_if_not_exists(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


def fillna_mean_bygroup(df, cols, group):
    cols = [col for col in cols if col.__contains__('#')]
    df[cols] = df.groupby(group)[cols].transform(lambda x: x.fillna(x.mean()))
    return df


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def get_samples_per_group(df, group):
    df_out = df.groupby(group).size()
    df_out.sort_values(inplace=True)
    df_out.to_csv(f'{tools.TOOLS_PATH}/samples_per_pid.csv')
