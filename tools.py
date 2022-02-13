import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import preprocess
import tools

RANDOM_SEED = 44
CAMPAIGN_4 = 4
CAMPAIGN_5 = 5
FEATURES_PATH = '/Users/aliceberg/Documents/Paper2022/data/FS_campaign_4_5-WIN_4_Jan.csv'
PREPROCESSED_FEATURES_PATH = 'data/cmp45_Jan/FS_campaign_4_5-WIN_4_preprocessed_Jan.csv'
PREPROCESSED_FEATURES_DEP_CMP5_PATH = 'data/cmp45_Jan/FS_campaign_4_5-WIN_4_preprocessed_dep_cmp5_Jan.csv'
PREPROCESSED_FEATURES_DEP_CMP4_PATH = 'data/cmp45_Jan/FS_campaign_4_5-WIN_4_preprocessed_dep_cmp4_Jan.csv'
PREPROCESSED_FEATURES_NONDEP_CMP5_PATH = 'data/cmp45_Jan/FS_campaign_4_5-WIN_4_preprocessed_nondep_cmp5_Jan.csv'
PREPROCESSED_FEATURES_NONDEP_CMP4_PATH = 'data/cmp45_Jan/FS_campaign_4_5-WIN_4_preprocessed_nondep_cmp4_Jan.csv'

CMP5_PRE_BDI_PATH = '/Users/aliceberg/Documents/Paper2022/docs/cmp5_preBDI.csv'
CMP4_PRE_BDI_PATH = '/Users/aliceberg/Documents/Paper2022/docs/cmp4_preBDI.csv'
TOOLS_PATH = 'tools/cmp45_Jan'
RESULTS_PATH = 'results'
SELECTED_PID_BY_PHQ_SAMPLES = [50114, 50125, 50288, 50396, 5049, 40105, 50175, 40137, 40110, 40126, 5079, 50379, 50139,
                               40103, 40125, 5097, 50360, 40119, 40116, 40108, 50224, 5060, 50455, 40150, 5057, 40122,
                               50119, 50358, 40115, 40178, 50454, 50229, 40134, 50255, 40179, 40170, 40169, 5096, 40132,
                               5014, 50106, 40114, 5021, 40109, 5073, 5061, 40143, 50134, 50400, 50163]
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
                           ]

DROP_PARTICIPANT_LIST = [4082, 4084, 4096]

SELECTED_PID_CMP4 = [40149, 40103, 40153, 40105, 40110, 40125, 40116, 40178, 40134, 40137, 40119, 4093, 40122, 40108,
                     40126, 40150, 40109, 40143, 40151, 40179, 4090
                     ]

SELECTED_PID_CMP5 = [
    5077, 50125, 50704, 50200, 50506, 50149, 50175, 50288, 50360, 50235, 50631, 50396, 50379, 50454, 5049,
    50114, 50692, 50737, 5079, 5057, 5039, 5021, 50480, 50688, 50472, 50580, 50715, 5097, 50224, 50157, 50588,
    50229, 5014, 50358, 50284, 50139, 50455, 50198, 50765, 50618, 50292, 50576, 50546, 5073, 50573, 50251, 50380,
    50586, 50466, 50667, 50119, 5078, 50163, 50471, 50687, 50248, 50351, 50426, 50405, 50483, 50579, 5060, 50255, 50624,
    50400, 5061, 50574, 50134, 50253, 50388, 50137, 50655, 50577, 50664, 50694, 5095, 50335, 50540, 50106, 50381, 50598,
    50620, 50593, 50221, 50318, 50236, 50409, 50646, 50356, 5071, 5067, 50597, 5084, 50583, 50470, 50502, 50129, 50698,
    50729, 50558, 50136, 5069
] # removed 50281, 50523, 50559, 50769 50741, 50737

SELECTED_PID_CMP4_DEP = [40149, 40103, 40153, 40105, 40110, 40125, 40116, 40178, 40134, 40137, 40119, 4093, 40122]

SELECTED_PID_CMP4_NON_DEP = [40114, 40133, 40174, 40171, 40164, 40141, 40163]

SELECTED_PID_CMP5_DEP = [50281, 50523, 50559, 50769, 50741, 50737,5077, 50704, 50200, 50175, 50288, 50360, 50235, 50396, 50379, 50454, 5049, 50114,
                         50692, 5057, 5039, 5021, 50480, 50472, 50580, 50715, 5097, 50157, 50588, 50229,
                         5014, 50358, 50455, 50198, 50618, 50292, 50576, 50546, 5073, 50573, 50380]

SELECTED_PID_CMP5_NON_DEP = [50125, 50506, 50149, 50631, 5079, 50688, 50224, 50284, 50139, 50251, 50119, 50687, 50483,
                             5060, 50255, 50624, 50134, 50253, 50577, 50694, 50106, 50221, 50409, 50356, 5084, 50583,
                             50470, 50698, 50742, 5052, 50180, 5065, 50344, 50604, 5096, 50262, 5062, 50194, 50408,
                             50103, 50222, 50652, 50463, 5081]

CMP4_DEP_PID_LIST = [4085, 4086, 4092, 4093, 4097, 4099, 40100, 40105, 40107, 40108, 40109, 40113, 40116, 40118, 40128,
                     40131, 40132, 40135, 40143, 40146, 40155, 40159, 40162, 40173, 40176, 40178, 40179, 4090, 40103,
                     40110, 40115, 40119, 40122, 40124, 40125, 40126, 40134, 40137, 40149, 40150, 40151, 40152, 40153,
                     40158, 40165, 40169, 40170]
CMP4_NON_DEP_PID_LIST = [4082, 4084, 4087, 4089, 4091, 4096, 4098, 40102, 40104, 40114, 40117,
                         40120, 40121, 40129, 40133, 40141, 40154, 40156, 40163, 40164, 40171,
                         40172, 40174]

CMP5_NON_DEP_PID_LIST = [50279, 50276, 50123, 50168, 50474, 50187, 50202, 50237, 50860, 50282, 50347, 50467, 50527,
                         50591, 50668, 50713, 50751, 50796, 50809, 50797, 50799, 50800, 50801, 50802, 50803, 50804,
                         50805, 50806, 50814, 50812, 50815, 50813, 50817, 50820, 50821, 50823, 50822, 50829, 50826,
                         50830, 50827, 50843, 50831, 50833, 50835, 50837, 50840, 50842, 50841, 50848, 50845, 50851,
                         50849, 50855, 50853, 50852, 50863, 5044, 50261, 50214, 50350, 50354, 50370, 50488, 50592,
                         50816, 50342, 50673, 50189, 50320, 50328, 50222, 50266, 50258, 50322, 50287, 50334, 50349,
                         50431, 50460, 50490, 50572, 50611, 50656, 50727, 50818, 5083, 50304, 50116, 5089, 5063, 50479,
                         50105, 50156, 50227, 50314, 50327, 50199, 50212, 50220, 50353, 50301, 50436, 50340, 50345,
                         50366, 50355, 50359, 50363, 50406, 50629, 50446, 50462, 50486, 50544, 50608, 50604, 50648,
                         50776, 50722, 50754, 50819, 50104, 5068, 50118, 50119, 50128, 50126, 50165, 50290, 50300, 5020,
                         5053, 50201, 50171, 50238, 50193, 50433, 50333, 50317, 50428, 50478, 50685, 50735, 50750,
                         50807, 5050, 50206, 5046, 50254, 50166, 50176, 50245, 50185, 50194, 50196, 50253, 50387, 50284,
                         50449, 50404, 50391, 50415, 50421, 50441, 50463, 50470, 50482, 50581, 50619, 50637, 50678,
                         50743, 50838, 50138, 5040, 5076, 50145, 5036, 5098, 50162, 50173, 50244, 50218, 50191, 50375,
                         50398, 50564, 50536, 50590, 50649, 50672, 50719, 50740, 50742, 50789, 50759, 50825, 50139,
                         5075, 5065, 50121, 50361, 50149, 5047, 5042, 50332, 50160, 50234, 50362, 50408, 50411, 50505,
                         50513, 50609, 50585, 50594, 50631, 50710, 50753, 50766, 50828, 50232, 50174, 5051, 50274, 5084,
                         50151, 50216, 50273, 50484, 50376, 50434, 50394, 50427, 50271, 50296, 50336, 50445, 50444,
                         50483, 50520, 50568, 50624, 50652, 50732, 50846, 50417, 50110, 5072, 50181, 50397, 50144,
                         50154, 50109, 50228, 50177, 50341, 50319, 50270, 50256, 50280, 50262, 50382, 50399, 50409,
                         50413, 50423, 50476, 50584, 50560, 50566, 50567, 50587, 50657, 50669, 50687, 50755, 50834,
                         50106, 50215, 50251, 50134, 50155, 50224, 50393, 5086, 50326, 50302, 50310, 50330, 50311,
                         50420, 50451, 50315, 50255, 50412, 50485, 50475, 50545, 50596, 50607, 50640, 50632, 50625,
                         50688, 50698, 50793, 50745, 50734, 50824, 50103, 50260, 50102, 5092, 5052, 5041, 5088, 5037,
                         50277, 50338, 50246, 50344, 50602, 50410, 50459, 50653, 50647, 50720, 50756, 50808, 50856,
                         50150, 50443, 50195, 50210, 50205, 50221, 50243, 50331, 50294, 50307, 50285, 50286, 50356,
                         50369, 50622, 50577, 50643, 50638, 50694, 50661, 50705, 50749, 5060, 5062, 50132, 5096, 5081,
                         50180, 50440, 5079, 5082, 5099, 50125, 50325, 5045, 50124, 50143, 50432, 50250, 50230, 50324,
                         50263, 50275, 50435, 50293, 50493, 50506, 50531, 50603, 50583, 50571, 50644, 50695, 50726,
                         50706, 50733, 50746, 50795,
                         ]

CMP5_DEP_PID_LIST = [50298, 50183, 50148, 5080, 50153, 50179, 5028, 50346, 50115, 50186, 50239, 50240, 50229, 50297,
                     50372, 50418, 50429, 50419, 50508, 50468, 50500, 50530, 50540, 50543, 50623, 50642, 50646, 50693,
                     50768, 50299, 506, 50761, 5074, 50146, 50131, 50197, 5030, 50312, 50555, 50295, 50414, 50430,
                     50395, 50471, 50472, 50526, 50561, 50606, 50601, 50635, 50747, 5066, 50133, 50163, 50323, 50247,
                     50337, 50368, 50511, 50522, 50580, 50576, 50614, 50651, 50683, 50700, 50716, 50729, 5069, 5070,
                     50101, 50252, 50213, 50313, 50257, 50371, 50519, 50634, 50675, 50670, 50739, 50771, 50309, 5067,
                     50111, 50425, 50288, 50501, 50600, 50365, 50407, 50589, 50549, 50680, 50762, 50778, 50267, 5058,
                     5095, 5094, 50241, 50226, 50422, 50348, 50377, 50492, 50613, 50541, 50524, 50641, 50691, 50662,
                     50665, 50728, 50738, 50744, 50792, 5097, 50396, 5055, 50122, 50158, 50188, 50207, 50248, 50597,
                     50380, 50464, 50452, 50494, 50497, 50509, 50557, 50748, 50836, 5021, 50236, 5071, 5091, 5056,
                     50127, 5073, 50137, 5048, 50113, 5049, 50335, 50367, 50401, 50465, 50502, 50655, 50666, 50736,
                     50757, 50839, 50844, 50850, 50233, 50264, 50141, 50268, 50405, 50242, 50269, 50272, 50593, 50379,
                     50447, 50487, 50558, 50618, 50598, 50586, 50633, 50654, 50741, 50810, 50832, 50854, 50278, 5090,
                     50249, 50223, 50424, 50306, 50374, 50381, 50388, 50481, 50538, 50599, 50610, 50702, 5059, 50152,
                     50456, 50203, 50450, 50454, 50562, 50559, 50636, 50630, 50684, 50767, 5087, 50510, 50308, 50426,
                     50305, 50461, 50477, 50498, 50535, 50548, 50553, 50650, 50737, 5054, 5026, 50161, 50343, 50400,
                     50605, 50570, 50588, 50621, 50628, 50689, 50709, 50704, 50752, 50772, 50219, 50265, 5014, 50231,
                     50292, 50595, 50690, 50692, 50711, 50769, 50198, 50466, 50569, 50563, 50616, 50626, 50721, 50100,
                     50129, 50142, 50135, 50147, 50316, 50167, 50575, 50674, 50723, 50130, 5093, 5078, 50318, 50351,
                     50403, 50514, 50533, 50579, 50658, 50701, 50660, 50708, 5085, 50225, 50697, 50717, 50715, 50763,
                     50774, 50798, 50235, 50114, 50458, 50550, 50659, 50770, 50787, 50208, 50358, 50758, 50281, 50291,
                     50352, 50620, 50764, 50765, 5061, 50861, 50455, 50546, 50107, 50170, 50489, 50480, 50521, 50786,
                     50259, 50172, 50389, 50469, 50542, 50532, 50573, 50760, 50157, 50437, 50718, 50788, 50439, 50574,
                     5043, 50178, 50360, 50117, 50664, 50790, 5057, 5064, 50175, 50416, 50556, 50378, 50448, 50667,
                     50453, 50582, 50707, 5077, 5039, 50329, 50525, 50682, 50639, 50730, 50303, 50612, 50373,
                     50507, 50136, 50200, 50617, 50402]


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


def remove_ds_store(filenames):
    if '.DS_Store' in filenames:
        filenames.remove('.DS_Store')
    return filenames


def split_file_by_campaigns(filename):
    df = pd.read_csv(filename)

    for campaign_id in tqdm([CAMPAIGN_4, CAMPAIGN_5]):
        df_cmp = df[df['pid'].astype(str).str.startswith(f'{campaign_id}0')]
        df_cmp.sort_values(by=['var_sum', 'samples'], inplace=True, ascending=False)

        filename_out = f"{filename.split('.')[0]}_cmp{campaign_id}.csv"
        df_cmp.to_csv(filename_out, index=False)


def preprocess_bdi_file(filename):
    df = pd.read_csv(filename)
    df['label'] = np.where(df['SUM'] < 14, 0, 1)
    df.to_csv(filename)
    return df


def split_file_by_depr_groups(filename=PREPROCESSED_FEATURES_PATH):
    df = pd.read_csv(filename)

    df_dep_5 = df[df['pid'].isin(CMP5_DEP_PID_LIST)]
    df_dep_4 = df[df['pid'].isin(CMP4_DEP_PID_LIST)]
    df_nondep_5 = df[df['pid'].isin(CMP5_NON_DEP_PID_LIST)]
    df_nondep_4 = df[df['pid'].isin(CMP4_NON_DEP_PID_LIST)]

    df_dep_4.to_csv(PREPROCESSED_FEATURES_DEP_CMP4_PATH)
    df_dep_5.to_csv(PREPROCESSED_FEATURES_DEP_CMP5_PATH)
    df_nondep_4.to_csv(PREPROCESSED_FEATURES_NONDEP_CMP4_PATH)
    df_nondep_5.to_csv(PREPROCESSED_FEATURES_NONDEP_CMP5_PATH)


def create_ema_stats_file():
    frames = []
    df = pd.read_csv(tools.PREPROCESSED_FEATURES_NONDEP_CMP5_PATH)
    for symptom in tools.SYMPTOM_ORIGINAL_COLUMN_LIST:
        frames.append(preprocess.get_feature_variation_per_participant(df, symptom))
    df_out = pd.concat(frames, axis=1)
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]

    df_out['var_sum'] = df_out["negative_self_image"] + df_out["sleep_trouble"] + df_out["depressed_feeling"] + df_out[
        "suicide_thoughts"] + df_out["difficulty_focusing"] + df_out["poor_appetite"] + df_out["lack_of_interest"] + \
                        df_out["fatigue"] + \
                        df_out["bad_physchomotor_activity"]

    df_out.reset_index(inplace=True)

    samples = []
    for row in df_out.itertuples():
        pid = row.pid
        df_pid = df[df['pid'] == pid]
        samples.append(len(df_pid))
    df_out['samples'] = samples
    df_out.sort_values(by=['var_sum', 'samples'], inplace=True, ascending=False)
    df_out.to_csv("tools/cmp45_Jan/combined_stdev_nondep_5.csv", index=False)


def subtract_lists(l1, l2):
    return [x for x in l1 if x not in l2]
