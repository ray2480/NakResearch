import copy
hop_length = 512
bpm = 90

background_percussive, sr = librosa.load('01.isft.background.perc.wav')
onset_envelope = librosa.onset.onset_strength(background_percussive, sr=sr, hop_length=hop_length)
onset_frames = librosa.util.peak_pick(onset_envelope, 7, 7, 7, 7, 0.8, 5)
beat_frames_per_16th_note = trackBeatsPer16thNote(background_percussive, bpm, sr=sr, hop_length=hop_length, offset_16th_notes=0)
quantized_onset_frames_per_16th_note, onset_frames_index_of_16th_notes = quantizeOnsetFramesPer16thNote(onset_frames, beat_frames_per_16th_note)
strong_onset_frames, strong_onset_frames_index_of_16th_notes = getStrongOnsetFrames(onset_envelope, beat_frames_per_16th_note, onset_frames_index_of_16th_notes)
weak_onset_frames, weak_onset_frames_index_of_16th_notes = getWeakOnsetFrames(strong_onset_frames_index_of_16th_notes, onset_frames_index_of_16th_notes, beat_frames_per_16th_note)

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
  return list(map(int, quantized_onset_frames_per_16th_note)), onset_frames_index_of_16th_notes

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
        #while onset_frames_index_of_16th_notes_removal[i+1] - onset_frames_index_of_16th_notes_removal[i] <= 1: #同じフレームをとってしまっているものは全消去
       #   del onset_frames_index_of_16th_notes_removal[i+1]
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
  return list(set(strong_onset_frames)).sort(), list(set(strong_onset_frames_index_of_16th_notes)).sort() #sortの必要あり
  #return strong_onset_frames, strong_onset_frames_index_of_16th_notes

def getWeakOnsetFrames(strong_onset_frames_index_of_16th_notes, onset_frames_index_of_16th_notes, beat_frames_per_16th_note):
  weak_onset_frames = []
  onset_frames_index_of_16th_notes_removal  = copy.deepcopy(onset_frames_index_of_16th_notes)
  #for strong_index in strong_onset_frames_index_of_16th_notes:
  i = 0
  while i < len(strong_onset_frames_index_of_16th_notes):
    del onset_frames_index_of_16th_notes_removal[strong_onset_frames_index_of_16th_notes[i]]
    i = i + 1
  weak_onset_frames_index_of_16th_notes = onset_frames_index_of_16th_notes_removal
  for weak_index in weak_onset_frames_index_of_16th_notes:
    weak_onset_frames.append(beat_frames_per_16th_note[weak_index])
  return weak_onset_frames, weak_onset_frames_index_of_16th_notes
  
