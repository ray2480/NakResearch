from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa
import copy
import mido

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
    S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))

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
    # harmonicとpercussiveに分離
    S_harmonic, S_percussive = librosa.decompose.hpss(D)
    return librosa.istft(S_harmonic), librosa.istft(S_percussive)

# 楽曲のビートを16分単位でトラッキングする
# 返戻値 beat_frames_per_16th_note は16分音符毎にビートを記録したフレームのnd.array（正確には4分音符単位でトラックしたものを4分割したもの）
def trackBeatsPer16thNote(x, bpm, sr=22050, hop_length=512, offset_16th_notes=0):
    """
    clickで書き出すと16分音符毎にクリック音が鳴らせるようにビートトラッキングする
    16分音符毎にインデックスが割り当てられている
    offset_16th_notes（最初の16分音符の数）でアウフタクトの除去が可能
    """
    tempo, beat_samples = librosa.beat.beat_track(x, sr=sr, hop_length=hop_length, start_bpm=bpm, units='samples')
    beat_frames_per_16th_note = []
    for i in range(len(beat_samples) - 1):
        interval_per_16th_units = librosa.samples_to_frames(np.linspace(beat_samples[i], beat_samples[i + 1], 5),
                                                            hop_length=hop_length)
        beat_frames_per_16th_note = np.hstack((beat_frames_per_16th_note, interval_per_16th_units))
    if offset_16th_notes > 0:
        beat_frames_per_16th_note = beat_frames_per_16th_note[offset_16th_notes:]
    return beat_frames_per_16th_note.astype(np.int)

# trackBeatsPer16thNoteでビートトラックをした後に得たbeat_frames_per_16th_note(16分音符毎のビート、4分音符毎に推定したビートを4分割したもの）を使って
# peak_pickで得たonset_framesをbeat_frames_per_16th_note画基準の16分音符単位にクォンタイズ(ticksが固定の全くテンポの揺れないピアノロールでのクォンタイズではない）
def quantizeOnsetFramesPer16thNote(onset_frames, beat_frames_per_16th_note):
    onset_frames_per_16th_note = []
    onset_frames_index_of_16th_notes = []
    for onset in onset_frames:
        # onset_framesとbeat_frames_per_16th_note(16分音符の位置)の差を取って, それが最小となる箇所のbeat_frames_per_16th_noteのインデックスを取得
        index = np.argmin(np.absolute(beat_frames_per_16th_note - onset))
        # onset_framesが16分音符でどの位置にあるかを示すインデックス群をonset_frames_index_of_16th_notesに格納
        # onset_frames_index_of_16th_notesはbeat_frames_per_16th_noteにおけるonset_framesが入る筈のインデックス情報を格納
        onset_frames_index_of_16th_notes.append(index)
        # onset_framesが16分音符でどの位置にあるかを示す
        onset_frames_per_16th_note.append(beat_frames_per_16th_note[index])
    # index取得の過程で出来たインデックスの重複を無くして時系列（フレーム(インデックス)の小さい順）にソートしたものを返す(list型)
    return sorted(list(set(list(map(int, onset_frames_per_16th_note))))), sorted(
        list(set(onset_frames_index_of_16th_notes)))

# 強リズムとなるようなonset framesを取得
def getStrongOnsetFrames(onset_envelope, beat_frames_per_16th_note, onset_frames_index_of_16th_notes):

    strong_onset_frames = [] # 強リズムとなるようなonset frames
    strong_onset_frames_index_of_16th_notes = [] # strong_onset_framesの値が入っているonset_framesのインデックス（onset_frames_index_of_16th_notes）
    # strong_onset_frames_index_of_16th_notesは小節ごとに16分音符が全部入っている箱のどこにstrong_onset_framesの情報が入るべきかという
    # ものでなくてはいけないはず だからonsetのインデックスのどこに入るかっていうのはちょっと違うと思われる
    onset_frames_index_of_16th_notes_copy = copy.deepcopy(onset_frames_index_of_16th_notes) # onset_frames_index_of_16th_notesのコピー(元をdelで壊さないようにするため）
    i = 0 # whileループのカウンター

    # onset_framesの数（インデックスの数）- 1 だけ走査（ -1 は onset_frames_index_of_16th_notes_copyのdiffを取るため）
    while i < len(onset_frames_index_of_16th_notes_copy) - 1:
        # onset_framesのインデックス（onset_frames_index_of_16th_notes）の差をとる事で16分音符何個分離れているかがわかるので、それによって処理を分ける
        diff_onset_frames_index_of_16th_notes = onset_frames_index_of_16th_notes_copy[i + 1] - onset_frames_index_of_16th_notes_copy[i]
        if diff_onset_frames_index_of_16th_notes <= 1:  # 16分音符が隣り合っている(1)（, 同じフレームをとってしまっている(0)（基本起こりえない筈））
            # onset_envelopeを比較 16分音符が隣り合っているものを比較し、onset_envelopeの大きさが大きい方をstrong_onset_framesに採用
            if onset_envelope[beat_frames_per_16th_note[onset_frames_index_of_16th_notes_copy[i]]] >= onset_envelope[
                beat_frames_per_16th_note[onset_frames_index_of_16th_notes_copy[i + 1]]]:
                index = onset_frames_index_of_16th_notes_copy[i]
                strong_onset_frames.append(beat_frames_per_16th_note[index])
                strong_onset_frames_index_of_16th_notes.append(index)
                del onset_frames_index_of_16th_notes_copy[i + 1]
            else:
                index = onset_frames_index_of_16th_notes_copy[i + 1]
                strong_onset_frames.append(beat_frames_per_16th_note[onset_frames_index_of_16th_notes_copy[i + 1]])
                strong_onset_frames_index_of_16th_notes.append(index)
                del onset_frames_index_of_16th_notes_copy[i]
        else:  # 16分音符が隣り合わない（8分以上の間隔がある)場合はonset_framesをそのままstrong_onset_framesとして採用
            index = onset_frames_index_of_16th_notes_copy[i]
            strong_onset_frames.append(beat_frames_per_16th_note[onset_frames_index_of_16th_notes_copy[i]])
            strong_onset_frames_index_of_16th_notes.append(index)
        i = i + 1 # whileループのカウントをインクリメント
    return strong_onset_frames, strong_onset_frames_index_of_16th_notes

# 弱リズムとなるようなonset framesを取得
def getWeakOnsetFrames(strong_onset_frames_index_of_16th_notes, onset_frames_index_of_16th_notes,
                       beat_frames_per_16th_note):

    weak_onset_frames = [] # 弱リズムとなるようなonset frames
    onset_frames_index_of_16th_notes_copy = copy.deepcopy(onset_frames_index_of_16th_notes) # onset_frames_index_of_16th_notesのコピー(元をremoveで壊さないようにするため）
    i = 0 # whileループのカウンター

    #strong_onset_framesの(インデックスの）数だけ走査
    while i < len(strong_onset_frames_index_of_16th_notes):
        # onset_framesからstrong_onset_framesのインデックスを削除
        onset_frames_index_of_16th_notes_copy.remove(strong_onset_frames_index_of_16th_notes[i])
        i = i + 1

    weak_onset_frames_index_of_16th_notes = onset_frames_index_of_16th_notes_copy
    for weak_index in weak_onset_frames_index_of_16th_notes:
        weak_onset_frames.append(beat_frames_per_16th_note[weak_index])
    return weak_onset_frames, weak_onset_frames_index_of_16th_notes

# MIDIデータの作成
def createMidiRhythmScore(midi_filename, onset_frames_index_of_16th_notes, strong_onset_frames_index_of_16th_notes,
                          weak_onset_frames_index_of_16th_notes, bpm, auftakt_16th_notes_number=0):

    ticks_per_16th_note = 120 # 16分音符のtickでの単位あたりの長さ
    ticks_per_beat = ticks_per_16th_note * 4  # 4分音符の場合ticksは480がデフォルト

    # 各音符の配置される場所（ticksで格納)
    onset_ticks = np.array(onset_frames_index_of_16th_notes) * ticks_per_16th_note #×16分音符の場所全部　〇onsetの場所全部
    strong_onset_ticks = np.array(strong_onset_frames_index_of_16th_notes) * ticks_per_16th_note
    weak_onset_ticks = np.array(weak_onset_frames_index_of_16th_notes) * ticks_per_16th_note

    # auftaktの処理（本来mido自体をいじるべきだが便宜上ここで）
    # onset_ticks = list(filter(lambda x: x >= ticks_per_16th_note * auftakt_16th_notes_number, onset_ticks))
    # strong_onset_ticks = list(filter(lambda x: x >= ticks_per_16th_note * auftakt_16th_notes_number, strong_onset_ticks))
    # weak_onset_ticks = list(filter(lambda x: x >= ticks_per_16th_note * auftakt_16th_notes_number, weak_onset_ticks))

    # MidiFileの初期設定
    smf = mido.MidiFile(ticks_per_beat=ticks_per_beat) # MidiFile(midiを扱うためのもの)インスタンスを生成
    track = mido.MidiTrack() # MidiFileの中身を記述するもの
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm))) # テンポを設定
    track.append(mido.Message('program_change', program=1))  # 音色を設定

    # 音符入力
    # auftaktの処理 16分音符分のアウフタクトを全音符から引いた長さを休符（delta time(note_off))として扱う
    # 楽譜の記譜上はアウフタクトにはなってないがこれで擬似的に表現
    track.append(
        mido.Message('note_off', time=(ticks_per_16th_note * 16) - (ticks_per_16th_note * auftakt_16th_notes_number)))
    i = 0
    for i in range(len(onset_ticks) - 1):
        delta = onset_ticks[i + 1] - onset_ticks[i]
        if onset_ticks[i] in strong_onset_ticks:
            track.append(mido.Message('note_on', velocity=100, note=librosa.note_to_midi('F3')))
            track.append(mido.Message('note_off', time=delta))
            track.append(mido.Message('note_off', note=librosa.note_to_midi('F3')))
        elif onset_ticks[i] in weak_onset_ticks:
            track.append(mido.Message('note_on', velocity=50, note=librosa.note_to_midi('A3')))
            track.append(mido.Message('note_off', time=delta))
            track.append(mido.Message('note_off', note=librosa.note_to_midi('A3')))

    track.append(mido.MetaMessage('end_of_track'))
    smf.tracks.append(track)
    # midiの出力
    smf.save(midi_filename)
