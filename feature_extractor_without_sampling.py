import matplotlib.pyplot as plt
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioAnalysis
from pyAudioAnalysis import ShortTermFeatures
from pydub import AudioSegment
import os
import numpy as np

mp3_dir = "dataset//DEAM_audio//MEMD_audio"
wav_dir = "dataset//DEAM_audio//wav_audio"
csv_dir = "dataset//extracted_features//no_sampling"
csv_file = csv_dir + "//features_no_sampling.csv"

#check if the directories are already there
if not os.path.exists(wav_dir):
    os.mkdir(wav_dir)
if not os.path.exists(csv_dir):
    os.mkdir(csv_dir)

# convert mp3 to wav
with os.scandir(mp3_dir) as dir:
    for song in dir:
        if song.is_file() and song.name.endswith(".mp3"):
            song_id = song.name.rstrip(".mp3")
            if not os.path.exists(wav_dir + song_id + ".wav"):
                sound = AudioSegment.from_mp3(song)
                sound.export(wav_dir + "//" + song_id + ".wav", format="wav")
                print("created " + song_id + ".wav")


song_size = 45
window_size = 500e-3

#perform audio feature extraction using ShortTermFeatures:
#output: a signle csv file where each row is a song, column 1 is the song id, and column 2 - 35 are features
with os.scandir(wav_dir) as dir:
    for song in dir:
        if song.is_file() and song.name.endswith(".wav"):
            sample_rate, signal = audioBasicIO.read_audio_file(song)
            #features_and_deltas should be an array with 1 column and 68 rows, the first 34 of them being the features we want
            features_and_deltas, feature_names = ShortTermFeatures.feature_extraction(signal, 2, window_size, window_size) #TODO: make this work
            features = features_and_deltas[:34,:].flatten()

            song_id = song.name.rstrip(".wav")
            with open(file=csv_file , mode='w', encoding='ASCII') as file:
                file.write(song_id)
                for feature in features:
                    file.write('/t' + feature)
                file.write('\n')