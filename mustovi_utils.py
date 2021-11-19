import musicnn
from musicnn.extractor import extractor
from musicnn.tagger import top_tags
import pandas as pd
import numpy as np
import librosa
import torch
#remove deprecation warnings
#import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def get_taggram(resampled_path, input_overlap, input_length):
    #MTT_musicnn', 'MTT_vgg', 'MSD_musicnn', 'MSD_musicnn_big' or 'MSD_vgg'.
    try:
        taggram_msd, tags_msd = extractor(resampled_path, model='MSD_musicnn_big', extract_features=False, 
                                          input_overlap=input_overlap, input_length=input_length)
    except ValueError:
        print("Please install musicnn from source to use MSD_musicnn_big. Defaulting to small model...")
        taggram_msd, tags_msd = extractor(resampled_path, model='MSD_musicnn', extract_features=False, 
                                          input_overlap=input_overlap, input_length=input_length)
    taggram_mtt, tags_mtt = extractor(resampled_path, model='MTT_musicnn', extract_features=False, 
                                      input_overlap=input_overlap, input_length=input_length)
    # clear cuda
    from numba import cuda
    cuda.select_device(0)
    cuda.close()
    # merge taggrams
    tag_df_msd = pd.DataFrame(taggram_msd, columns=tags_msd)
    tag_df_mtt = pd.DataFrame(taggram_mtt, columns=tags_mtt)
    tag_df = pd.concat([tag_df_msd, tag_df_mtt], axis=1)
    # merge beat (before normalization because we merged them before calculating the stds)
    tag_df["Beat"] = (tag_df["beat"] + tag_df["beats"]) / 2
    tag_df = tag_df.drop(columns=["beat", "beats"])
    # identify duplicated columns
    duplicated_cols = set()
    for col in tag_df.columns:
        if isinstance(tag_df[col], pd.DataFrame):
            duplicated_cols.add(col)
    # average duplicated columns
    for duplicated_col in list(duplicated_cols):
        averaged = tag_df[duplicated_col].mean(axis=1)
        del tag_df[duplicated_col]
        tag_df[duplicated_col] = averaged
    # normalize taggram by std on the MTT dataset
    mtt_stds = pd.read_csv("data/mtt/mtt_stds.csv", header=None, index_col=0, squeeze=True).iloc[1:]
    normed_tag_df = tag_df / mtt_stds
    # merge some more columns
    merge_dict = {("male vocal", "male voice", "male vocalists"): "male vocal",
                  ("female vocal", "female voice", "female vocalists"): "female vocal",
                  ("vocal", "vocals", "voice"): "vocal",
                  ("no vocal", "no vocals", "no voice"): "no vocal",
                  ("electro", "electronic"): "electro",
                  ("choir", "choral"): "choral", 
                 }
    print(normed_tag_df.columns)
    for cols_to_merge in merge_dict:
        cols_to_merge_list = list(cols_to_merge)
        merged_col = normed_tag_df[cols_to_merge_list].to_numpy().mean(axis=1)
        normed_tag_df = normed_tag_df.drop(columns=cols_to_merge_list)
        normed_tag_df[merge_dict[cols_to_merge]] = merged_col
    return normed_tag_df


def get_spec_norm(song):
    mel_spec = librosa.feature.melspectrogram(song, sr=16000, S=None, 
                                              n_fft=512, hop_length=256, 
                                              win_length=None, window='hann', center=True, 
                                              pad_mode='reflect', power=2.0)
    # Obtain maximum value per time-frame
    spec_max = np.amax(mel_spec, axis=0)
    #print(spec_max.shape)
    # Normalize all values between 0 and 1
    mel_spec = (mel_spec - np.min(spec_max)) / np.ptp(spec_max)

    #mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
    #sns.heatmap(mel_spec[:20, :])
    return mel_spec.mean(axis=0)


def slerp(low, high, val):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    epsilon = 1e-7
    omega = (low_norm * high_norm).sum(1)
    omega = torch.acos(torch.clamp(omega, -1 + epsilon, 1 - epsilon))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res