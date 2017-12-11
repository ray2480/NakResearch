

tempo, beat_times = librosa.beat.beat_track(x, sr=sr, start_bpm=90, units='time')
#アウフタクトの除去
beat_times_omitted = beat_times[1:]

#16分刻みのクォンタイズ値を生成
beat_times_sixteenths = []
for i in range(len(beat_times_omitted) - 1):
  sixteenths_space = np.linspace(beat_times_omitted[i], beat_times_omitted[i+1], 5)
  beat_times_sixteenths = np.hstack((beat_times_sixteenths, sixteenths_space))
