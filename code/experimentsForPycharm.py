from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa
import copy
import mido
from functions import separateMusicIntoForegroundAndBackground, separateMusicIntoHarmonicAndPercussive,\
    trackBeatsPer16thNote, quantizeOnsetFramesPer16thNote, getStrongOnsetFrames, getWeakOnsetFrames, createMidiRhythmScore

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
auftakt_16th_notes_number_array = [2, 8, 0, 0, 0, 0, 0, 0, 0, 0]

if __name__ == '__main__':
    ##background_percussiveからクリック音を原曲に入れる
    i = 0
    for i in range(len(input_audio_filename_array)):
        peak_pick_delta = 0.8
        input_audio_filename = input_audio_filename_array[i]
        x, sr = librosa.load(input_audio_filename)
        audio_filename_background_percussive = input_audio_filename + '_background_percussive.wav'
        back, sr = librosa.load(audio_filename_background_percussive)
        bpm = input_bpm_array[i]
        auftakt_16th_notes_number = auftakt_16th_notes_number_array[i]

        # ビート処理
        onset_envelope = librosa.onset.onset_strength(back, sr=sr, hop_length=hop_length)
        # onset_frames = librosa.util.peak_pick(onset_envelope, 7, 7, 7, 7, 0.8, 5)
        onset_frames = librosa.util.peak_pick(onset_envelope, 7, 7, 7, 7, peak_pick_delta, 5)
        beat_frames_per_16th_note = trackBeatsPer16thNote(back, bpm, sr=sr, hop_length=hop_length, offset_16th_notes=0)
        onset_frames_per_16th_note, onset_frames_index_of_16th_notes = quantizeOnsetFramesPer16thNote(
            onset_frames, beat_frames_per_16th_note)
        strong_onset_frames, strong_onset_frames_index_of_16th_notes = getStrongOnsetFrames(onset_envelope,
                                                                                            beat_frames_per_16th_note,
                                                                                            onset_frames_index_of_16th_notes)
        weak_onset_frames, weak_onset_frames_index_of_16th_notes = getWeakOnsetFrames(
            strong_onset_frames_index_of_16th_notes, onset_frames_index_of_16th_notes, beat_frames_per_16th_note)

        # クリック音入力
        N = len(x)
        T = N / float(sr)
        clicks_strong = librosa.clicks(frames=strong_onset_frames, sr=sr, hop_length=hop_length, length=N)
        clicks_weak = librosa.clicks(frames=weak_onset_frames, sr=sr, hop_length=hop_length, click_freq=1000.0,
                                     click_duration=0.01, length=N)

        # audio_filename_clicks = input_audio_filename + '_clicks.wav'
        audio_filename_clicks = input_audio_filename + '_clicks' + str(peak_pick_delta) + '.wav'
        librosa.output.write_wav(audio_filename_clicks, x + clicks_strong + clicks_weak, sr)

        # リズム譜作成
        midi_filename = input_audio_filename + '.mid'
        createMidiRhythmScore(midi_filename, onset_frames_index_of_16th_notes, strong_onset_frames_index_of_16th_notes,
                              weak_onset_frames_index_of_16th_notes, bpm, auftakt_16th_notes_number)


def createMidiFromAudioData():
    i = 0
    for i in range(len(input_audio_filename_array)):
        input_audio_filename = input_audio_filename_array[i]
        bpm = input_bpm_array[i]
        auftakt_16th_notes_number = auftakt_16th_notes_number_array[i]

        # foreground, background, background_percussiveの出力ファイル名
        audio_filename_foreground = input_audio_filename + '_foreground.wav'
        audio_filename_background = input_audio_filename + '_background.wav'
        audio_filename_background_percussive = input_audio_filename + '_background_percussive.wav'

        # 入力wavデータ読み込み、分離処理
        x, sr = librosa.load(input_audio_filename)
        S_foreground, S_background = separateMusicIntoForegroundAndBackground(x, sr)
        background_harmonic, background_percussive = separateMusicIntoHarmonicAndPercussive(S_background)

        # foreground, background, background_percussiveのwavファイル書き出し
        librosa.output.write_wav(audio_filename_foreground, librosa.istft(S_foreground), sr)
        librosa.output.write_wav(audio_filename_background, librosa.istft(S_background), sr)
        librosa.output.write_wav(audio_filename_background_percussive, background_percussive, sr)

        # ビート処理
        onset_envelope = librosa.onset.onset_strength(background_percussive, sr=sr, hop_length=hop_length)
        onset_frames = librosa.util.peak_pick(onset_envelope, 7, 7, 7, 7, 0.8, 5)
        beat_frames_per_16th_note = trackBeatsPer16thNote(background_percussive, bpm, sr=sr, hop_length=hop_length,
                                                          offset_16th_notes=0)
        quantized_onset_frames_per_16th_note, onset_frames_index_of_16th_notes = quantizeOnsetFramesPer16thNote(
            onset_frames, beat_frames_per_16th_note)
        strong_onset_frames, strong_onset_frames_index_of_16th_notes = getStrongOnsetFrames(onset_envelope,
                                                                                            beat_frames_per_16th_note,
                                                                                            onset_frames_index_of_16th_notes)
        weak_onset_frames, weak_onset_frames_index_of_16th_notes = getWeakOnsetFrames(
            strong_onset_frames_index_of_16th_notes, onset_frames_index_of_16th_notes, beat_frames_per_16th_note)

        # クリック音入力
        N = len(background_percussive)
        T = N / float(sr)
        # t = np.linspace(0, T, len(onset_envelope))
        clicks_strong = librosa.clicks(frames=strong_onset_frames, sr=sr, hop_length=hop_length, length=N)
        clicks_weak = librosa.clicks(frames=weak_onset_frames, sr=sr, hop_length=hop_length, click_freq=1000.0,
                                     click_duration=0.01, length=N)
        audio_filename_background_percussive_clicks = audio_filename_background_percussive + '_clicks.wav'
        librosa.output.write_wav(audio_filename_background_percussive_clicks,
                                 background_percussive + clicks_strong + clicks_weak, sr)

        # リズム譜作成
        midi_filename = input_audio_filename + '.mid'
        createMidiRhythmScore(midi_filename, onset_frames_index_of_16th_notes, strong_onset_frames_index_of_16th_notes,
                              weak_onset_frames_index_of_16th_notes, bpm, auftakt_16th_notes_number)
