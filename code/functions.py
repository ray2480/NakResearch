
def beatTrackPer16th(x, sr, bpm, units, offset_16th_notes):
  tempo, beat_times = librosa.beat.beat_track(x, sr=sr, start_bpm=90, units='time')
  #アウフタクトの除去
  #beat_times = beat_times[1:]
  #16分刻みのクォンタイズ値を生成
  beat_times_per_16th = []
  for i in range(len(beat_times) - 1):
    interval_per_16th_units = np.linspace(beat_times[i], beat_times[i+1], 5)
    beat_times_per_16th = np.hstack((beat_times_per_16th, interval_per_16th_units))
  return beat_times_per_16th
