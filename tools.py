import os

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
DROP_PARTICIPANT_LIST = [4082, 4084, 4096]


def create_dir_if_not_exists(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


def fillna_mean_bygroup(df, cols, group):
    cols = [col for col in cols if col.__contains__('#')]
    df[cols] = df.groupby(group)[cols].transform(lambda x: x.fillna(x.mean()))
    return df
