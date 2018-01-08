from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa
import copy
import mido

hop_length = 512
#1曲目
bpm = 90
auftakt_16th_notes_number = 6
#2曲目
input_audio_filename = '02.Track_2.wav'
bpm = 100
auftakt_16th_notes_number = 0
#3曲目
input_audio_filename = '03.Track_3.wav'
bpm = 108
auftakt_16th_notes_number = 0
#4曲目
input_audio_filename = '06.Track_6.wav'
bpm = 120
auftakt_16th_notes_number = 4
#5曲目
input_audio_filename = '08.Track_8.wav'
bpm = 127
auftakt_16th_notes_number = 0
#6曲目
input_audio_filename = '12.Track_12.wav'
bpm = 121
auftakt_16th_notes_number = 0
#7曲目
input_audio_filename = '13.Track_13.wav'
bpm = 103
auftakt_16th_notes_number = 0
#8曲目
input_audio_filename = '14.Track_14.wav'
bpm = 88
auftakt_16th_notes_number = 0
#9曲目
input_audio_filename = 'M02_01.Track_1.wav'
bpm = 97
auftakt_16th_notes_number = 0
#10曲目
input_audio_filename = 'M02_03.Track_3.wav'
bpm = 130
auftakt_16th_notes_number = 0
#11曲目
input_audio_filename = 'M02_02.Track_2.wav'
bpm = 112
auftakt_16th_notes_number = 0
#12曲目
input_audio_filename = 'M02_06.Track_6.wav'
bpm = 135
auftakt_16th_notes_number = 0

input_audio_filename_array = ['03.Track_3.wav', '06.Track_6.wav', '08.Track_8.wav', '12.Track_12.wav', '13.Track_13.wav',
                              '14.Track_14.wav', 'M02_01.Track_1.wav', 'M02_03.Track_3.wav', 'M02_02.Track_2.wav', 'M02_06.Track_6.wav']
input_bpm_array = [108, 120, 127, 121, 103, 88, 97, 130, 112, 135]
auftakt_16th_notes_number_array = [0, 4, 0, 0, 0, 0, 0, 0, 0, 0]

#foreground, background, background_percussiveの出力ファイル名
audio_filename_foreground = input_audio_filename + '_foreground.wav'
audio_filename_background = input_audio_filename + '_background.wav'
audio_filename_background_percussive = input_audio_filename + '_background_percussive.wav'

#入力wavデータ読み込み、分離処理
x, sr = librosa.load(input_audio_filename)
S_foreground, S_background = separateMusicIntoForegroundAndBackground(x, sr)
background_harmonic, background_percussive = separateMusicIntoHarmonicAndPercussive(S_background)

#foreground, background, background_percussiveのwavファイル書き出し
librosa.output.write_wav(audio_filename_foreground, librosa.istft(S_foreground), sr)
librosa.output.write_wav(audio_filename_background, librosa.istft(S_background), sr)
librosa.output.write_wav(audio_filename_background_percussive, background_percussive, sr)

#ビート処理
onset_envelope = librosa.onset.onset_strength(background_percussive, sr=sr, hop_length=hop_length)
onset_frames = librosa.util.peak_pick(onset_envelope, 7, 7, 7, 7, 0.8, 5)
beat_frames_per_16th_note = trackBeatsPer16thNote(background_percussive, bpm, sr=sr, hop_length=hop_length, offset_16th_notes=0)
quantized_onset_frames_per_16th_note, onset_frames_index_of_16th_notes = quantizeOnsetFramesPer16thNote(onset_frames, beat_frames_per_16th_note)
strong_onset_frames, strong_onset_frames_index_of_16th_notes = getStrongOnsetFrames(onset_envelope, beat_frames_per_16th_note, onset_frames_index_of_16th_notes)
weak_onset_frames, weak_onset_frames_index_of_16th_notes = getWeakOnsetFrames(strong_onset_frames_index_of_16th_notes, onset_frames_index_of_16th_notes, beat_frames_per_16th_note)

#クリック音入力
N = len(background_percussive)
T = N/float(sr)
#t = np.linspace(0, T, len(onset_envelope))
clicks_strong = librosa.clicks(frames=strong_onset_frames, sr=sr, hop_length=hop_length, length=N)
clicks_weak = librosa.clicks(frames=weak_onset_frames, sr=sr, hop_length=hop_length, click_freq=1000.0, click_duration=0.01, length=N)
audio_filename_background_percussive_clicks = audio_filename_background_percussive + '_clicks.wav'
librosa.output.write_wav(audio_filename_background_percussive_clicks, background_percussive+clicks_strong+clicks_weak, sr)

#リズム譜作成
midi_filename = input_audio_filename + '.mid'
createMidiRhythmScore(midi_filename, onset_frames_index_of_16th_notes, strong_onset_frames_index_of_16th_notes, weak_onset_frames_index_of_16th_notes, bpm, auftakt_16th_notes_number)


def separateMusicIntoForegroundAndBackground(x, sr):
  # compute the spectrogram magnitude and phase
  S_full, phase = librosa.magphase(librosa.stft(x))

  # We'll compare frames using cosine similarity, and aggregate similar frames
  # by taking their (per-frequency) median value.
  #
  # To avoid being biased by local continuity, we constrain similar frames to be
  # separated by at least 2 seconds.
  #
  # This suppresses sparse/non-repetetitive deviations from the average spectrum,
  # and works well to discard vocal elements.
  S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine', width=int(librosa.time_to_frames(2, sr=sr)))

  # The output of the filter shouldn't be greater than the input
  # if we assume signals are additive.  Taking the pointwise minimium
  # with the input spectrum forces this.
  S_filter = np.minimum(S_full, S_filter)

  # We can also use a margin to reduce bleed between the vocals and instrumentation masks.
  # Note: the margins need not be equal for foreground and background separation
  margin_i, margin_v = 2, 10
  power = 2
  mask_i = librosa.util.softmask(S_filter, margin_i * (S_full - S_filter), power=power)
  mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)

  # Once we have the masks, simply multiply them with the input spectrum
  # to separate the components
  S_foreground = mask_v * S_full
  S_background = mask_i * S_full
  
  return S_foreground, S_background

def separateMusicIntoHarmonicAndPercussive(D):
  #harmonicとpercussiveに分離
  S_harmonic, S_percussive = librosa.decompose.hpss(D)
  return librosa.istft(S_harmonic), librosa.istft(S_percussive)

def trackBeatsPer16thNote(x, bpm, sr=22050, hop_length=512, offset_16th_notes=0):
  """
  clickで書き出すと16分音符毎にクリック音が鳴らせるようにビートトラッキングする
  16分音符毎にインデックスが割り当てられている
  offset_16th_notes（最初の16分音符の数）でアウフタクトの除去が可能
  """
  tempo, beat_samples = librosa.beat.beat_track(x, sr=sr, hop_length=hop_length, start_bpm=bpm, units='samples')
  beat_frames_per_16th_note = []
  for i in range(len(beat_samples ) - 1):
    interval_per_16th_units = librosa.samples_to_frames(np.linspace(beat_samples[i], beat_samples[i+1], 5), hop_length=hop_length)    
    beat_frames_per_16th_note = np.hstack((beat_frames_per_16th_note, interval_per_16th_units))
  if offset_16th_notes > 0:
    beat_frames_per_16th_note = beat_frames_per_16th_note[offset_16th_notes:]
  return beat_frames_per_16th_note.astype(np.int)

def quantizeOnsetFramesPer16thNote(onset_frames, beat_frames_per_16th_note):
  quantized_onset_frames_per_16th_note = []
  onset_frames_index_of_16th_notes = []
  for onset in onset_frames:
    index = np.argmin(np.absolute(beat_frames_per_16th_note - onset))
    onset_frames_index_of_16th_notes.append(index)
    quantized_onset_frames_per_16th_note.append(beat_frames_per_16th_note[index])
  return sorted(list(set(list(map(int, quantized_onset_frames_per_16th_note))))), sorted(list(set(onset_frames_index_of_16th_notes)))

def getStrongOnsetFrames(onset_envelope, beat_frames_per_16th_note, onset_frames_index_of_16th_notes):
  #本当にdelでちゃんと次の音符飛ばしが出来ているかちゃんと検証する必要あり（もしかしたらそうなってないかもしれない）
  strong_onset_frames = []
  strong_onset_frames_index_of_16th_notes = []
  onset_frames_index_of_16th_notes_removal = copy.deepcopy(onset_frames_index_of_16th_notes)
  i = 0
  while i < len(onset_frames_index_of_16th_notes_removal) - 1:
    diff_onset_frames_index_of_16th_notes = onset_frames_index_of_16th_notes_removal[i+1] - onset_frames_index_of_16th_notes_removal[i]
    if diff_onset_frames_index_of_16th_notes <= 1: #16分音符が隣り合っている(1), 同じフレームをとってしまっている(0)
      if onset_envelope[beat_frames_per_16th_note[onset_frames_index_of_16th_notes_removal[i]]] >= onset_envelope[beat_frames_per_16th_note[onset_frames_index_of_16th_notes_removal[i+1]]]:
        index = onset_frames_index_of_16th_notes_removal[i]
        strong_onset_frames.append(beat_frames_per_16th_note[index])
        strong_onset_frames_index_of_16th_notes.append(index)
        del onset_frames_index_of_16th_notes_removal[i+1]
      else:
        index = onset_frames_index_of_16th_notes_removal[i+1]
        strong_onset_frames.append(beat_frames_per_16th_note[onset_frames_index_of_16th_notes_removal[i+1]])
        strong_onset_frames_index_of_16th_notes.append(index)
        del onset_frames_index_of_16th_notes_removal[i]
    else: #16分音符が隣り合わない（8分以上の間隔がある)
      index = onset_frames_index_of_16th_notes_removal[i]
      strong_onset_frames.append(beat_frames_per_16th_note[onset_frames_index_of_16th_notes_removal[i]])
      strong_onset_frames_index_of_16th_notes.append(index)
    i = i + 1
  return strong_onset_frames, strong_onset_frames_index_of_16th_notes

def getWeakOnsetFrames(strong_onset_frames_index_of_16th_notes, onset_frames_index_of_16th_notes, beat_frames_per_16th_note):
  weak_onset_frames = []
  onset_frames_index_of_16th_notes_removal  = copy.deepcopy(onset_frames_index_of_16th_notes)
  i = 0
  while i < len(strong_onset_frames_index_of_16th_notes):
    onset_frames_index_of_16th_notes_removal.remove(strong_onset_frames_index_of_16th_notes[i])
    i = i + 1
  weak_onset_frames_index_of_16th_notes = onset_frames_index_of_16th_notes_removal
  for weak_index in weak_onset_frames_index_of_16th_notes:
    weak_onset_frames.append(beat_frames_per_16th_note[weak_index])
  return weak_onset_frames, weak_onset_frames_index_of_16th_notes

def createMidiRhythmScore(midi_filename, onset_frames_index_of_16th_notes, strong_onset_frames_index_of_16th_notes, weak_onset_frames_index_of_16th_notes, bpm, auftakt_16th_notes_number=0):
  #MIDIデータの作成
  #16分音符のtickでの単位あたりの長さ
  ticks_per_16th_note = 120
  ticks_per_beat = ticks_per_16th_note * 4 #4分音符は480がデフォルト
  #各音符の配置される場所（tick)
  onset_ticks = np.array(onset_frames_index_of_16th_notes) * ticks_per_16th_note
  strong_onset_ticks = np.array(strong_onset_frames_index_of_16th_notes) * ticks_per_16th_note
  weak_onset_ticks = np.array(weak_onset_frames_index_of_16th_notes) * ticks_per_16th_note
  #auftaktの処理（本来mido自体をいじるべきだが便宜上ここで）
  #onset_ticks = list(filter(lambda x: x >= ticks_per_16th_note * auftakt_16th_notes_number, onset_ticks))
  #strong_onset_ticks = list(filter(lambda x: x >= ticks_per_16th_note * auftakt_16th_notes_number, strong_onset_ticks))
  #weak_onset_ticks = list(filter(lambda x: x >= ticks_per_16th_note * auftakt_16th_notes_number, weak_onset_ticks))
  #事前処理
  smf = mido.MidiFile(ticks_per_beat=ticks_per_beat)
  track = mido.MidiTrack()
  track.append(mido.MetaMessage('set_tempo',tempo=mido.bpm2tempo(bpm)))
  track.append(mido.Message('program_change', program=1)) #音色 
  #音符入力
  #最初だけデルタタイム入れる
  onset_ticks_diff = np.diff(onset_ticks)
  #auftaktの処理
  #track.append(mido.Message('note_off',time=(ticks_per_16th_note * 12))) 
  track.append(mido.Message('note_off',time=(ticks_per_16th_note * 16 ) - (ticks_per_16th_note * auftakt_16th_notes_number))) 
  i = 0
  for i in range(len(onset_ticks) - 1):
    delta = onset_ticks[i+1] - onset_ticks[i]
    if onset_ticks[i] in strong_onset_ticks:
      track.append(mido.Message('note_on', velocity=100, note=librosa.note_to_midi('F3')))
      track.append(mido.Message('note_off',time=delta)) 
      track.append(mido.Message('note_off',note=librosa.note_to_midi('F3')))
    elif onset_ticks[i] in weak_onset_ticks:
      track.append(mido.Message('note_on', velocity=50, note=librosa.note_to_midi('A3')))
      track.append(mido.Message('note_off',time=delta)) 
      track.append(mido.Message('note_off',note=librosa.note_to_midi('A3')))
  track.append(mido.MetaMessage('end_of_track'))
  smf.tracks.append(track)
  #midiの出力
  smf.save(midi_filename)
  
  
def midi():
  if(小節の頭or３拍め):
    コードを追加、セット
  if (strong):
    root音入れる
  if (weak) :
    5th入れる
  
#def createMidiForBacking(strong_onset_frames, weak_onset_frames):
def 
"""
16分音符4個か8個単位で区切ってコードを入力
コードのリストを入力すると４分か２分音符単位でstrong, weakに沿って打ち込んでく

"""
