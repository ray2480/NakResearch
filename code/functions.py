#clickで書き出すと16分音符毎にクリック音が鳴らせるようにビートトラッキングする
#16分音符毎にインデックスが割り当てられている
#beat_times_per_16thの値は秒数、サンプル、フレームのどれかで選べる
def beatTrackPer16th(x, sr, bpm, units='frames', offset_16th_notes=0):
  if units == 'time':
    tempo, beat_times = librosa.beat.beat_track(x, sr=sr, start_bpm=bpm, units='time')
  if units == 'samples':
    tempo, beat_times = librosa.beat.beat_track(x, sr=sr, start_bpm=bpm, units='samples')
  else:
    tempo, beat_times = librosa.beat.beat_track(x, sr=sr, start_bpm=bpm, units='frames')
  beat_times_per_16th = []
  for i in range(len(beat_times) - 1):
    interval_per_16th_units = np.linspace(beat_times[i], beat_times[i+1], 5)
    beat_times_per_16th = np.hstack((beat_times_per_16th, interval_per_16th_units))
  if offset_16th_notes > 0:
    #アウフタクトの除去
    beat_times_per_16th = beat_times_per_16th[offset_16th_notes:]
  return beat_times_per_16th

def quantizeOnsetFramesBeatsToIntempo16Beats(onset_frames, beat_times_per_16th, sr, hop_length)
  beat_times_omitted_quantized = []
  sixteenths_index = []
  for onset in librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length):
    index = np.argmin(np.absolute(beat_times_sixteenths - onset))
    sixteenths_index.append(index)
    beat_times_omitted_quantized.append(beat_times_sixteenths[index])
  sixteenths_index = np.array(sixteenths_index)
  
  return intempo_16beats_quantized_beats, sixteenths_index
