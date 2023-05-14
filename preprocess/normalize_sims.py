# *_*coding:utf-8 *_*
"""
normalize sims: only for audio and visual features
"""

import os
import pickle
import numpy as np

from config.get_data_root import data_root


sims_path = os.path.join(data_root, 'Datasets/SIMS/Processed/unaligned_39.pkl')
data = pickle.load(open(sims_path, 'rb'))

# normalize audio and video
eps = 1e-6
train_audio_mean, train_audio_std = None, None
train_video_mean, train_video_std = None, None
for split in ['train', 'valid', 'test']:
    # audio
    audio_data = data[split]['audio']
    audio_lengths = data[split]['audio_lengths']
    num_samples = audio_data.shape[0]
    audio_packed = np.concatenate([audio_data[i, :audio_lengths[i]] for i in range(num_samples)])
    # video
    video_data = data[split]['vision']
    video_lengths = data[split]['vision_lengths']
    video_packed = np.concatenate([video_data[i, :video_lengths[i]] for i in range(num_samples)])
    # get train mean & std
    if split == 'train':
        train_audio_mean, train_audio_std = np.mean(audio_packed, axis=0), np.std(audio_packed, axis=0)
        train_video_mean, train_video_std = np.mean(video_packed, axis=0), np.std(video_packed, axis=0)
    # normalize
    for i in range(num_samples):
        a_len, v_len = audio_lengths[i], video_lengths[i]
        audio_data[i, :a_len] = (audio_data[i, :a_len] - train_audio_mean) / (train_audio_std + eps)
        video_data[i, :v_len] = (video_data[i, :v_len] - train_video_mean) / (train_video_std + eps)

# save
sims_path_normalized = os.path.join(data_root, 'Datasets/SIMS/Processed/unaligned_39_normalized.pkl')
pickle.dump(data, open(sims_path_normalized, 'wb'))
print(f"==>: Normalized sims data is saved at '{sims_path_normalized}'")



