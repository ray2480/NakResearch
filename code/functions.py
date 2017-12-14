import copy
import mido
hop_length = 512
bpm = 90
auftakt_16th_notes_number = 6

background_percussive, sr = librosa.load('01.isft.background.perc.wav')
onset_envelope = librosa.onset.onset_strength(background_percussive, sr=sr, hop_length=hop_length)
onset_frames = librosa.util.peak_pick(onset_envelope, 7, 7, 7, 7, 0.8, 5)
beat_frames_per_16th_note = trackBeatsPer16thNote(background_percussive, bpm, sr=sr, hop_length=hop_length, offset_16th_notes=0)
quantized_onset_frames_per_16th_note, onset_frames_index_of_16th_notes = quantizeOnsetFramesPer16thNote(onset_frames, beat_frames_per_16th_note)
strong_onset_frames, strong_onset_frames_index_of_16th_notes = getStrongOnsetFrames(onset_envelope, beat_frames_per_16th_note, onset_frames_index_of_16th_notes)
weak_onset_frames, weak_onset_frames_index_of_16th_notes = getWeakOnsetFrames(strong_onset_frames_index_of_16th_notes, onset_frames_index_of_16th_notes, beat_frames_per_16th_note)

N = len(background_percussive)
T = N/float(sr)
t = np.linspace(0, T, len(onset_envelope))
clicks_strong = librosa.clicks(frames=strong_onset_frames, sr=sr, hop_length=hop_length, length=N)
clicks_weak = librosa.clicks(frames=weak_onset_frames, sr=sr, hop_length=hop_length, click_freq=1000.0, click_duration=0.01, length=N)
output_filename = '01.Track_1.background_percussive.strong_and_weak_onset_frames.wav'
librosa.output.write_wav(output_filename, background_percussive+clicks_strong+clicks_weak, sr)

#リズム譜作成
midi_filename = '01_beat_from_createMidiRhythmScore.mid'
createMidiRhythmScore(midi_filename, onset_frames_index_of_16th_notes, strong_onset_frames_index_of_16th_notes, weak_onset_frames_index_of_16th_notes, bpm, auftakt_16th_notes_number)

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
  track.append(mido.Message('note_off',time=(ticks_per_16th_note * 12))) 
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
