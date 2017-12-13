hop_length = 512

onset_envelope = librosa.onset.onset_strength(background_percussive, sr=sr, hop_length=hop_length)
onset_frames = librosa.util.peak_pick(onset_envelope, 7, 7, 7, 7, 0.8, 5)

def trackBeatsPer16thNote(x, bpm, sr=22050, hop_length=512, units='frames', offset_16th_notes=0):
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
  beat_frames_per_16th_note = []
  for i in range(len(beat_times) - 1):
    interval_per_16th_units = np.linspace(beat_times[i], beat_times[i+1], 5)
    beat_frames_per_16th_note = np.hstack((beat_frames_per_16th_note, interval_per_16th_units))
  if offset_16th_notes > 0:
    beat_frames_per_16th_note = beat_frames_per_16th_note[offset_16th_notes:]
  return beat_frames_per_16th_note

def quantizeOnsetFramesPer16thNote(onset_frames, beat_frames_per_16th_note):
  quantized_onset_frames_per_16th_note = []
  onset_frames_index_of_16th_notes = []
  for onset in onset_frames:
    index = np.argmin(np.absolute(beat_frames_per_16th_note - onset))
    onset_frames_index_of_16th_notes.append(index)
    quantized_onset_frames_per_16th_note.append(beat_frames_per_16th_note[index])
  return quantized_onset_frames_per_16th_note, onset_frames_index_of_16th_notes

def getStrongOnsetFrames(onset_envelope, beat_frames_per_16th_note, onset_frames_index_of_16th_notes):
  #本当にdelでちゃんと次の音符飛ばしが出来ているかちゃんと検証する必要あり（もしかしたらそうなってないかもしれない）
  strong_onset_frames = []
  strong_onset_frames_index_of_16th_notes = []
  i = 0
  for i in range(len(onset_frames_index_of_16th_notes) - 1):
    if onset_frames_index_of_16th_notes[i+1] - onset_frames_index_of_16th_notes[i] == 1: #16分音符が隣り合っている
      if onset_envelope[beat_frames_per_16th_note[onset_frames_index_of_16th_notes[i]]] >= onset_envelope[beat_frames_per_16th_note[onset_frames_index_of_16th_notes[i+1]]]:
        index = onset_frames_index_of_16th_notes[i]
        strong_onset_frames.append(beat_frames_per_16th_note[index])
        strong_onset_frames_index_of_16th_notes.append(index)
        del onset_frames_index_of_16th_notes[i+1]
      else:
        index = onset_frames_index_of_16th_notes[i+1]
        strong_onset_frames.append(beat_frames_per_16th_note[onset_frames_index_of_16th_notes[i+1]])
        strong_onset_frames_index_of_16th_notes.append(index)
        del onset_frames_index_of_16th_notes[i]
    else:
      index = onset_frames_index_of_16th_notes[i]
      strong_onset_frames.append(beat_frames_per_16th_note[onset_frames_index_of_16th_notes[i]])
      strong_onset_frames_index_of_16th_notes.append(index)
  return strong_onset_frames, strong_onset_frames_index_of_16th_notes

def getWeakOnsetFrames(strong_onset_frames_index_of_16th_notes, onset_frames_index_of_16th_notes, beat_frames_per_16th_note):
  weak_onset_frames = []
  for strong_index in strong_onset_frames_index_of_16th_notes:
    del onset_frames_index_of_16th_notes[strong_index]
  weak_onset_frames_index_of_16th_notes = onset_frames_index_of_16th_notes
  for weak_index in weak_onset_frames_index_of_16th_notes:
    weak_onset_frames.append(beat_frames_per_16th_note[weak_index])
  return weak_onset_frames, weak_onset_frames_index_of_16th_notes
  
