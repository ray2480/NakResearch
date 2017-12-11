hop_length = 256


onset_envelope = librosa.onset.onset_strength(background_percussive, sr=sr, hop_length=hop_length)
onset_frames = librosa.util.peak_pick(onset_envelope, 7, 7, 7, 7, 0.8, 5)

def beatTrackPer16th(x, bpm, sr=22050, hop_length=512, units='frames', offset_16th_notes=0):
  """
  clickで書き出すと16分音符毎にクリック音が鳴らせるようにビートトラッキングする
  16分音符毎にインデックスが割り当てられている
  beat_frames_per_16thの値は秒数、サンプル、フレームのどれかで選べる
  offset_16th_notes（最初の16分音符の数）でアウフタクトの除去が可能
  """
  if units == 'time':
    tempo, beat_times = librosa.beat.beat_track(x, sr=sr, hop_length=hop_length, start_bpm=bpm, units='time')
  if units == 'samples':
    tempo, beat_times = librosa.beat.beat_track(x, sr=sr, hop_length=hop_length, start_bpm=bpm, units='samples')
  else:
    tempo, beat_times = librosa.beat.beat_track(x, sr=sr, hop_length=hop_length, start_bpm=bpm, units='frames')
  beat_frames_per_16th = []
  for i in range(len(beat_times) - 1):
    interval_per_16th_units = np.linspace(beat_times[i], beat_times[i+1], 5)
    beat_frames_per_16th = np.hstack((beat_frames_per_16th, interval_per_16th_units))
  if offset_16th_notes > 0:
    beat_frames_per_16th = beat_frames_per_16th[offset_16th_notes:]
  return beat_frames_per_16th

def quantizeOnsetFramesPer16thNote(onset_frames, beat_frames_per_16th):
  quantized_onset_frames_per_16th_note = []
  index_of_16th_notes = []
  for onset in onset_frames:
    index = np.argmin(np.absolute(beat_frames_per_16th - onset))
    index_of_16th_notes.append(index)
    quantized_onset_frames_per_16th_note.append(beat_frames_per_16th[index])
  return quantized_onset_frames_per_16th_note, index_of_16th_notes

def getStrongOnsetFrames(onset_envelop, beat_frames_per_16th, index_of_16th_notes):
  #本当にdelでちゃんと次の音符飛ばしが出来ているかちゃんと検証する必要あり（もしかしたらそうなってないかもしれない）
  strong_onset_frames = []
  i = 0
  for i in range(len(index_of_16th_notes) - 1):
    if index_of_16th_notes[i+1] - index_of_16th_notes[i] == 1: #16分音符が隣り合っている
      if onset_envelop[beat_frames_per_16th[index_of_16th_notes[i]]] >= onset_envelop[beat_frames_per_16th[index_of_16th_notes[i+1]]]:
        strong_onset_frames.append(beat_frames_per_16th[index_of_16th_notes[i]])
        del index_of_16th_notes[i+1]
      else:
        strong_onset_frames.append(beat_frames_per_16th[index_of_16th_notes[i+1]])
        del index_of_16th_notes[i]
    else:
      strong_onset_frames.append(beat_frames_per_16th[index_of_16th_notes[i]])
  return strong_onset_frames
