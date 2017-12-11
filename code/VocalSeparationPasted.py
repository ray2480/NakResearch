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
#onset_frames = librosa.util.peak_pick(onset_envelope, 7, 7, 7, 7, 0.5, 5)
onset_frames = librosa.util.peak_pick(onset_envelope, 7, 7, 7, 7, 0.8, 5)

#検出したピークに縦線を描画(カラーバー）
plt.plot(t, onset_envelope)
plt.vlines(t[onset_frames], 0, onset_envelope.max(), color='r', alpha=0.7)
plt.xlabel('Time (sec)')
plt.xlim(0, T)
plt.ylim(0)

#検出したピークにクリック音を入れる
clicks = librosa.clicks(frames=onset_frames, sr=sr, hop_length=hop_length, length=N)

#クリック音と共に出力
#librosa.output.write_wav('01.Track_1.background_percussive_click.wav', background_percussive+clicks, sr)
librosa.output.write_wav('01.Track_1.background_percussive_click_delta08.wav', background_percussive+clicks, sr)

#RWC-MDB-P-2001-M06の01はbpm = 90なのでnumpyでarrayをつくる　やることは16分でクォンタイズ
#BPM90で１小節が2.666666...秒
#beat, tempoの取得
tempo, beat_times = librosa.beat.beat_track(x, sr=sr, start_bpm=90, units='time')
#アウフタクトの除去
beat_times_omitted = beat_times[1:]
#16分刻みのクォンタイズ値を生成
beat_times_sixteenths = []
for i in range(len(beat_times_omitted) - 1):
  sixteenths_space = np.linspace(beat_times_omitted[i], beat_times_omitted[i+1], 5)
  beat_times_sixteenths = np.hstack((beat_times_sixteenths, sixteenths_space))
#16分にクォンタイズ
beat_times_omitted_quantized = []
sixteenths_index = []
for onset in librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length):
  index = np.argmin(np.absolute(beat_times_sixteenths - onset))
  sixteenths_index.append(index)
  beat_times_omitted_quantized.append(beat_times_sixteenths[index])
sixteenths_index = np.array(sixteenths_index)
#出力
clicks = librosa.clicks(beat_times_omitted_quantized, sr=sr, length=len(background_percussive))
librosa.output.write_wav(' 01.Track_1.background_percussive_quantized.wav', background_percussive+clicks, sr)

##########################################################
 
#librosaの関数でnovelty functionのonset_envelop生成(hop_length違う）
hop_length = 256
onset_envelope = librosa.onset.onset_strength(background_percussive, sr=sr, hop_length=hop_length)

tempo, beat_times = librosa.beat.beat_track(x, sr=sr, start_bpm=90, units='time')
#アウフタクトの除去
beat_times_omitted = beat_times[1:]

#16分刻みのクォンタイズ値を生成
beat_times_sixteenths = []
for i in range(len(beat_times_omitted) - 1):
  sixteenths_space = np.linspace(beat_times_omitted[i], beat_times_omitted[i+1], 5)
  beat_times_sixteenths = np.hstack((beat_times_sixteenths, sixteenths_space))
  
#16分にクォンタイズ
i = 0
onset_frames = np.array([])
#onsetを16分単位で抽出
"""
for i in range(len(beat_times) - 1):
  wait = int(librosa.time_to_samples(beat_times[i+1] - beat_times[i],  sr=sr) / 4) #16分のサンプル数
  start_quarter_frame = int(librosa.time_to_frames(beat_times[i], sr=sr, hop_length=hop_length))
  end_quarter_frame = int(librosa.time_to_frames(beat_times[i+1], sr=sr, hop_length=hop_length))
  onset_envelope_part = onset_envelope[start_quarter_frame:end_quarter_frame+1]
  onset_frames_part = librosa.util.peak_pick(onset_envelope_part, 7, 7, 7, 7, 0.8, wait - 1) + end_quarter_frame
  j = 0
  #フレームがかぶってしまっているので同じ値を持っていたら（重複していたら）削除
  for j in range(len(onset_frames_part)):
    if onset_frames_part[j] in onset_frames:
      np.delete(onset_frames_part, j)
  onset_frames = np.hstack((onset_frames, onset_frames_part))
"""
#16分と8分のonset_framesを作成
wait_16 = int(librosa.time_to_frames(np.min(np.diff(beat_times_omitted))/4,sr=sr, hop_length=hop_length))
onset_frames_16 = librosa.util.peak_pick(onset_envelope, 7, 7, 7, 7, 0.5, wait_16)
wait_8 = int(librosa.time_to_frames(np.min(np.diff(beat_times_omitted))/2,sr=sr, hop_length=hop_length))
onset_frames_8 = librosa.util.peak_pick(onset_envelope, 7, 7, 7, 7, 0.5, wait_8)
"""
#onset_frames_8が表拍といってる訳じゃない 最初に来たものをとって8分でとばしてるだけだから裏拍を取ってる可能性は十分ありうる
#onset_frames_8の次に来る音符と音符の間に飛ばしてしまった音符、つまりonset_frames_weakがある（裏拍じゃない）
#onset_frames_weakはインデックスにすべき このインデックスを見つけたら前後どちらがonset_envelopにおいて強いか比較して
#大きい方を採用 このときonset_frames_weakを採用すると以降のonset_frames_8がずれることになるので
#もしonset_frames_weakを採用したら強配列にappendし、もう一度その位置から8分配列はpeak_pickをしていく（再帰的処理）
#いや再起処理はpeak pickに対してではなく16分配列を8分間隔のstepでとって行く（あとweakも同じく）
#それもいや、peak_pickをしないアプローチをとるならいったんインテンポの16分に全部アサインしてしまってしまえば同間隔でとりだせるから
#元のインデックスだけちゃんと保持してしまえば関係ないところは0埋め、8部の場所は８、16分のところは16みたいなvalueをつけた
#配列を作ってそっちで処理してしまえばなんとかなるのでは　そうすればずれにも対応できる
16分の配列をインデックスごとにまわす
16beats[i]
"""

###裏箔のonset_framesを作成
onset_frames_weak = onset_frames_16.tolist()
for onset_frame_8 in onset_frames_8:
  if onset_frame_8 in onset_frames_weak:# ***1
    onset_frames_weak_index = onset_frames_weak.index(onset_frame_8)
    del onset_frames_weak[onset_frames_weak_index]
onset_frames_weak = np.array(onset_frames_weak) #weakというか裏拍
#onset_frames_strong = onset_frames_8 嘘　8分の単位で入ってこない要素があるから
"""
裏拍の配列と16分の配列を比較
強配列を作成、音符の採用方法は8分のステップ毎に16分（裏拍）配列を見て飛ばしていく　
1. 裏拍配列の前に何もない→強拍配列にappend その後ろに表拍があるならそれを飛ばす
(←これやるなら裏拍配列の前後表拍は削除じゃなくて別の文字列とかにした方が良いのでは　というより16分での裏拍のインデックスを把握すればよいのでは）
2. 裏拍配列の前に表拍あり→前後のフレームとってonset_envelopの強度を比較して大きいほうを強に
→もし表が大きいなら表が強、もし裏が大きいなら表は飛ばして裏を強にして次の表拍をとばす

16分配列を1つずつ走査 for i in range(len(onset_frames_16)):
16分で次が続いてるかどうかの判断　

"""


#intempoな16分にクォンタイズ(midiにあわせるため）
"""
def quantizeOnsetFramesBeatsToIntempo16Beats(onset_frames, sr, hop_length)
  beat_times_omitted_quantized = []
  sixteenths_index = []
  for onset in librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length):
    index = np.argmin(np.absolute(beat_times_sixteenths - onset))
    sixteenths_index.append(index)
    beat_times_omitted_quantized.append(beat_times_sixteenths[index])
  sixteenths_index = np.array(sixteenths_index)
  
  return intempo_16beats_quantized_beats, sixteenths_index
"""
###########################################################

import mido
#MIDIデータの作成
_song_bpm = 90 #曲のbpm
_ticks_per_beat = 480 #デフォルト
auftakt_sixteenths_num = 6
_sixteenth_sec = beat_times_sixteenths[1] - beat_times_sixteenths[0] #単位あたりの16分音符の秒での長さ
#16分音符のtickでの単位あたりの長さ
#sixteenth_tick = int(mido.second2tick(_sixteenth_sec, ticks_per_beat=_ticks_per_beat, tempo=mido.bpm2tempo(_song_bpm)))
sixteenth_tick = 120
#各音符の配置される場所（tick)
beat_times_ticks = sixteenths_index * sixteenth_tick #omittedのほう採用してるから最初の６個は０になっちゃってる
beat_times_ticks_omitted = beat_times_ticks[auftakt_sixteenths_num-1:]
#事前処理
smf = mido.MidiFile(ticks_per_beat=_ticks_per_beat)
track = mido.MidiTrack()
track.append(mido.MetaMessage('set_tempo',tempo=mido.bpm2tempo(_song_bpm)))
track.append(mido.Message('program_change', program=1)) #音色 
#音符入力
#最初だけデルタタイム入れる
beat_times_ticks_omitted_diff = np.diff(beat_times_ticks_omitted)
track.append(mido.Message('note_off',time=beat_times_ticks_omitted[0])) 
for delta in beat_times_ticks_omitted_diff:
  track.append(mido.Message('note_on', velocity=100, note=librosa.note_to_midi('F3')))
  track.append(mido.Message('note_off',time=delta)) 
  track.append(mido.Message('note_off',note=librosa.note_to_midi('F3')))
track.append(mido.MetaMessage('end_of_track'))
smf.tracks.append(track)
#midiの出力
smf.save('01_beat.mid')
