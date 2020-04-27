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
import csv

#if there is error with audio library path to ffmpeg needs to be explicitly defined
#AudioSegment.converter = r"C:\\Users\\jorda\Documents\\Python\\ffmpeg-4.2.2-win64-static\\bin\\ffmpeg.exe"

# 1st step - convert files from mp3 to wav, example for one song below
src = "2.mp3"
dst = "2.wav"
mp3_dir = "dataset//DEAM_audio//MEMD_audio"
wav_dir = "dataset//DEAM_audio//wav_audio"
csv_dir = "dataset//extracted_features//sampling"
csv_file = csv_dir + "//features_no_sampling.csv"
#directory where dynamic annotations per frame are stored
dyn_ann_dir = "dataset//DEAM_annotations//annotations//dynamic//annotations averaged per song//dynamic (per second annotations)//"
#directory where static annotations per song are stored
stat_ann_dir = "dataset//DEAM_annotations//annotations//song_level//"

#check if the directories are already there
if not os.path.exists(wav_dir):
    os.mkdir(wav_dir)
if not os.path.exists(csv_dir):
    os.mkdir(csv_dir)

""" Performs batch conversion of mp3 files specified in mp3_dir to wav files to wav_dir"""
def convert_mp3_to_wav(mp3_dir, wav_dir):
    with os.scandir(mp3_dir) as dir:
        for song in dir:
            if song.is_file() and song.name.endswith(".mp3"):
                song_id = song.name.rstrip(".mp3")
                if not os.path.exists(wav_dir + song_id + ".wav"):
                    sound = AudioSegment.from_mp3(song)
                    sound.export(wav_dir + "//" + song_id + ".wav", format="wav")
                    print("created " + song_id + ".wav")


""" Performs batch feature extraction of wav files specified in wav directory
    for the number of specified frames
    Converts stereo to mono signal
    Saves extracted features into csv files in csv_dir.
    One csv files per song. Columns represent the features, rows - the frames.
    The first row is a header row with feature names
    The first column is the frame counter """
def extract_features_per_frame(wav_dir, csv_dir, nr_frames):              
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
                features_and_deltas, feature_names = ShortTermFeatures.feature_extraction(signal, sample_rate, len(signal)/nr_frames, len(signal)/nr_frames) #TODO: make this work
                print(len(features_and_deltas))
                print(features_and_deltas)
                features = np.transpose(features_and_deltas[:34,:]) # why is it limited to 34
                #print(features)
                song_id = song.name.rstrip(".wav")
                #with open(csv_dir+"//"+song_id+'.csv', 'w', newline='') as csvfile:
                    #writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    #writer.writerow([features])
                np.savetxt(fname=csv_dir+"//"+song_id+'.csv', X=features, delimiter=',', encoding='ASCII') # this does not save properly with delimiter
                #song can be loaded into an np array named loaded_array with
                #loaded_array = np.loadtxt(fname=csv_dir + '//' + song_id + '.csv', delimiter=',', encoding='ASCII')

""" Performs batch feature extraction of wav files specified in wav directory
    Converts stereo to mono signal
    Saves extracted features into a csv files in csv_dir, where row represents songs and columns features extracted"""
def extract_features_per_song(wav_dir, csv_file):              
    song_size = 45
    window_size = 500

    #perform audio feature extraction using ShortTermFeatures:
    #output: a single csv file that contains the song name and all 34 features for each song.
    with os.scandir(wav_dir) as dir:
        with open(file=csv_file, mode='w', encoding='ASCII') as file:
            for song in dir:
                if song.is_file() and song.name.endswith(".wav"):
                    print("Processing song: " + song.name)
                    sample_rate, signal = audioBasicIO.read_audio_file(song)
                    print('Sample rate: ', sample_rate)
                    print("Number of samples: ",len(signal))
                    signal = audioBasicIO.stereo_to_mono(signal)
                    #sample rate retrieved from read audio file is different that sample rate used as a second arg in feature extraction
                    features_and_deltas, feature_names = ShortTermFeatures.feature_extraction(signal, sample_rate, len(signal), len(signal))
                    print(len(features_and_deltas))
                    features = np.transpose(features_and_deltas[:34,:]).flatten()

                    song_id = song.name.rstrip(".wav")

                    writer = csv.writer(file, delimiter='\t')
                    writer.writerow([song_id, features])
                        #file.write(song_id)
                        #for feature in features:
                            #file.write('/t')
                            #file.write(str(feature))
                        #file.write('\n')

""" Performs merging csv files: features extracted per song and static annotations with arousal/valence per song
The resulting csv file has format:

songID  Feature1    Feature2    ... FeatureN    Arousal Valence"""

#def merge_features_annontations_per_song(features_dir, stat_ann_dir):

#convert_mp3_to_wav(mp3_dir, wav_dir)
#extract_features_per_frame(wav_dir, csv_dir, 90)
extract_features_per_song(wav_dir, csv_file)

#there should be two variants of feature extraction (in two separate scripts): one that uses frame size of 500 msec (sampling rate of 2 Hz) and feature extraction per song level - using entire song (45 sec) as a frame size


#in the first variant there will be one csv file per song with columns that correspond to features and rows to windows
#in the second script the output will be one csv file for features as columns and rows as songs
