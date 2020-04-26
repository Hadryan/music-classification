'''
Created on Apr 16, 2020

@author: kaytee


'''
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioAnalysis
from pyAudioAnalysis import ShortTermFeatures
from pydub import AudioSegment
import os
import numpy as np

#if there is error with audio library path to ffmpeg needs to be explicitly defined
#AudioSegment.converter = r"C:\\Users\\jorda\Documents\\Python\\ffmpeg-4.2.2-win64-static\\bin\\ffmpeg.exe"

# 1st step - convert files from mp3 to wav, example for one song below
src = "2.mp3"
dst = "2.wav"
mp3_dir = "dataset//DEAM_audio//MEMD_audio"
wav_dir = "dataset//DEAM_audio//wav_audio"
csv_dir = "dataset//extracted_features//sampling"

#check if the directories are already there
if not os.path.exists(wav_dir):
    os.mkdir(wav_dir)
if not os.path.exists(csv_dir):
    os.mkdir(csv_dir)

# convert mp3 to wav
def convert_mp3_to_wav(mp3_dir, wav_dir):
    with os.scandir(mp3_dir) as dir:
        for song in dir:
            if song.is_file() and song.name.endswith(".mp3"):
                song_id = song.name.rstrip(".mp3")
                if not os.path.exists(wav_dir + song_id + ".wav"):
                    sound = AudioSegment.from_mp3(song)
                    sound.export(wav_dir + "//" + song_id + ".wav", format="wav")
                    print("created " + song_id + ".wav")


def extract_features(wav_dir):              
    song_size = 45
    window_size = 500

    #perform audio feature extraction using ShortTermFeatures:
    #output: a csv file for each song where the title is the song ID, columns correspond to features and rows to windows
    with os.scandir(wav_dir) as dir:
        for song in dir:
            if song.is_file() and song.name.endswith(".wav"):
                print("Processing song: " + song.name)           
                sample_rate, signal = audioBasicIO.read_audio_file(song)
                print('Sample rate: ', sample_rate)
                print("Number of samples: ",len(signal))
                signal = audioBasicIO.stereo_to_mono(signal)
                #sample rate retrieved from read audio file is different that sample rate used as a second arg in feature extraction
                features_and_deltas, feature_names = ShortTermFeatures.feature_extraction(signal, sample_rate, len(signal)/90, len(signal)/90) #TODO: make this work
                print(len(features_and_deltas))
                features = np.transpose(features_and_deltas[:34,:])

                song_id = song.name.rstrip(".wav")
                np.savetxt(fname=song_id+'.csv', X=features, encoding='ASCII')
                #song can be loaded into an np array named loaded_array with
                #loaded_array = np.loadtxt(fname=csv_dir + song_id + '.csv',  encoding='ASCII')


    #print(signal.shape)

#convert_mp3_to_wav(mp3_dir, wav_dir)
extract_features(wav_dir)

#there should be two variants of feature extraction (in two separate scripts): one that uses frame size of 500 msec (sampling rate of 2 Hz) and feature extraction per song level - using entire song (45 sec) as a frame size

#in the first variant there will be one csv file per song with columns that correspond to features and rows to windows
#in the second script the output will be one csv file for features as columns and rows as songs
