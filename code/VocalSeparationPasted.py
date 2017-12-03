##Vocal separation##

# Code source: Brian McFee
# License: ISC

##################
# Standard imports
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa

import librosa.display

y, sr = librosa.load('audio/Cheese_N_Pot-C_-_16_-_The_Raps_Well_Clean_Album_Version.mp3', duration=120)


# And compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y))

# We'll compare frames using cosine similarity, and aggregate similar frames
# by taking their (per-frequency) median value.
#
# To avoid being biased by local continuity, we constrain similar frames to be
# separated by at least 2 seconds.
#
# This suppresses sparse/non-repetetitive deviations from the average spectrum,
# and works well to discard vocal elements.

S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))

# The output of the filter shouldn't be greater than the input
# if we assume signals are additive.  Taking the pointwise minimium
# with the input spectrum forces this.
S_filter = np.minimum(S_full, S_filter)

# We can also use a margin to reduce bleed between the vocals and instrumentation masks.
# Note: the margins need not be equal for foreground and background separation
margin_i, margin_v = 2, 10
power = 2

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

# Once we have the masks, simply multiply them with the input spectrum
# to separate the components

S_foreground = mask_v * S_full
S_background = mask_i * S_full

##audioに変換
foreground = librosa.istft(S_foreground)
background = librosa.istft(S_background)

#出力
librosa.output.write_wav(' 01.Track_1.foreground.wav', foreground, sr)
librosa.output.write_wav(' 01.Track_1.background.wav', background, sr)


#harmonicとpercussiveに分離
H_background, P_background = librosa.decompose.hpss(S_background)
S_background_harmonic, S_background_percussive = librosa.decompose.hpss(S_background)
background_percussive = librosa.istft(S_background_percussive)
librosa.output.write_wav(' 01.Track_1.background_percussive.wav', background_percussive, sr)

#rmse log energy novelty function
hop_length = 512
frame_length = 1024
rmse = librosa.feature.rmse(background_percussive, frame_length=frame_length, hop_length=hop_length).flatten()
log_rmse = np.log1p(10*rmse)
log_rmse_diff = np.zeros_like(log_rmse)
log_rmse_diff[1:] = np.diff(log_rmse)

log_energy_novelty = np.max([np.zeros_like(log_rmse_diff), log_rmse_diff], axis=0)

#グラフ表示
plt.figure(figsize=(15, 6))
plt.plot(t, log_rmse, 'b--', t, log_rmse_diff, 'g--^', t, log_energy_novelty, 'r-')
plt.xlim(0, t.max())
plt.xlabel('Time (sec)')
plt.legend(('log RMSE', 'delta log RMSE', 'log energy novelty')) 

#librosaの関数でnovelty functionのonset_envelop生成(hop_length違う）
hop_length = 256
onset_envelope = librosa.onset.onset_strength(background_percussive, sr=sr, hop_length=hop_length)

#onset_envelopの描画
N = len(background_percussive)
T = N/float(sr)
t = np.linspace(0, T, len(onset_envelope))

plt.plot(t, onset_envelope)
plt.xlabel('Time (sec)')
plt.xlim(xmin=0)
plt.ylim(0)

#peak_picking
onset_frames = librosa.util.peak_pick(onset_envelope, 7, 7, 7, 7, 0.5, 5)

#検出したピークに縦線を描画(カラーバー）
plt.plot(t, onset_envelope)
plt.vlines(t[onset_frames], 0, onset_envelope.max(), color='r', alpha=0.7)
plt.xlabel('Time (sec)')
plt.xlim(0, T)
plt.ylim(0)

#検出したピークにクリック音を入れる
clicks = librosa.clicks(frames=onset_frames, sr=sr, hop_length=hop_length, length=N)

#クリック音と共に出力
librosa.output.write_wav('01.Track_1.background_percussive_click.wav', background_percussive+clicks, sr)

