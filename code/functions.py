#clickで書き出すと16分音符毎にクリック音が鳴らせるようにビートトラッキングする
#16分音符毎にインデックスが割り当てられている
#beat_times_per_16thの値は秒数、サンプル、フレームのどれかで選べる
def beatTrackPer16th(x, sr, bpm, units='time', offset_16th_notes=0):
  if units == 'frames':
    tempo, beat_times = librosa.beat.beat_track(x, sr=sr, start_bpm=bpm, units='frames')
  if units == 'samples':
    tempo, beat_times = librosa.beat.beat_track(x, sr=sr, start_bpm=bpm, units='samples')
  else:
    tempo, beat_times = librosa.beat.beat_track(x, sr=sr, start_bpm=bpm, units='time')
  beat_times_per_16th = []
  for i in range(len(beat_times) - 1):
    interval_per_16th_units = np.linspace(beat_times[i], beat_times[i+1], 5)
    beat_times_per_16th = np.hstack((beat_times_per_16th, interval_per_16th_units))
  if offset_16th_notes > 0:
    #アウフタクトの除去
    beat_times_per_16th = beat_times_per_16th[offset_16th_notes:]
  return beat_times_per_16th


