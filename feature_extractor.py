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
csv_file = "dataset//extracted_features//features_no_sampling.csv"
#directory where dynamic annotations per frame are stored
dyn_ann_dir = "dataset//DEAM_annotations//annotations//dynamic//"
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
    One csv files per song, title is the song_id. Columns represent the features, rows - the frames.
    The first row is a header row with feature names
    The first column is the frame counter """
def extract_features_per_frame(wav_dir, csv_dir, nr_frames):
    #song_size = 45000
    #window_size = 500
    with os.scandir(wav_dir) as dir:
        for song in dir:
            song_id = song.name.rstrip(".wav")
            if song.is_file() and song.name.endswith(".wav") and not os.path.exists(csv_dir + '/' + song_id + '.csv'):
                print("Processing song: " + song.name)
                sample_rate, signal = audioBasicIO.read_audio_file(song)
                print('Sample rate: ', sample_rate)
                print("Number of samples: ",len(signal))
                #perform conversion from stereo to mono
                signal = audioBasicIO.stereo_to_mono(signal)
                #perform audio feature extraction using ShortTermFeatures:
                features_and_deltas, feature_names = ShortTermFeatures.feature_extraction(signal, sample_rate, len(signal)/nr_frames, len(signal)/nr_frames)
                features = np.transpose(features_and_deltas[:34,:])
                #output: a csv file for each song where the title is the song ID, columns correspond to features and rows to windows
                with open(file=csv_dir + '/' + song_id + '.csv', mode='w', encoding='ASCII') as file:
                    print("Writing " + csv_dir + '/' + song_id + '.csv')
                    file.write('frame #' + np_row_to_string(feature_names[:34]) + '\n')
                    current_frame = 0
                    for row in features:
                        file.write(str(current_frame) + np_row_to_string(row) + '\n')
                        current_frame += 1


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
            header_written = False
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

                    if not header_written:
                        file.write('song id' + np_row_to_string(feature_names[:34]) + '\n')
                        header_written = True

                    file.write(song_id + np_row_to_string(features) + '\n')




""" Converts a row from a numpy array to a string with the format
    ,element_1,element_2,...,element_n
    Used when printing to files"""
def np_row_to_string(row):
    result = ''
    for element in row:
        result += ',' + str(element)

    return result

""" Combs through all of the files in static_ann_dir and return a 2d numpy array where each row is
    song_id, arousal_mean, valence_mean"""
def combine_static_annotation(stat_ann_dir):
    result = []
    for file in os.scandir(stat_ann_dir):
        if file.is_file() and file.name.endswith('.csv'):
            print('Extracting from file',file.name)
            with open(file) as csvfile:
                reader = csv.reader(csvfile,skipinitialspace=True)
                header = next(reader)
                id_index = header.index('song_id')
                arousal_index = header.index('arousal_mean')
                valence_index = header.index('valence_mean')
                for row in reader:
                    result.append([row[id_index], row[arousal_index], row[valence_index]])

    return result


""" Performs merging csv files: csvs with features extracted per frame and dynamic annotations with arousal/valence per frame
The resulting csv file has format:

songId_frameID  Feature1    Feature2    ... FeatureN    Arousal Valence"""

#def merge_features_annontations_per_frame(features_dir, stat_ann_dir):

""" Performs merging csv files: features extracted per song and static annotations with arousal/valence per song
The resulting csv file has format:

songID  Feature1    Feature2    ... FeatureN    Arousal Valence"""

def merge_features_annontations_per_song(features_file, stat_ann_dir):
    extracted_features = np.loadtxt(fname=features_file,dtype=object,delimiter=',')
    annotations = combine_static_annotation(stat_ann_dir)

    output_file = features_file[:-4] + '_merge.csv'

    with open(output_file, 'w') as output:
        output.write(extracted_features[0][0] + np_row_to_string(extracted_features[0][1:]) + ',arousal,valence\n')

        for i in range(1,extracted_features.shape[0]):
            song_id = extracted_features[i][0]
            row_to_write = extracted_features[i][0] + np_row_to_string(extracted_features[i][1:])
            #not the fastest way of searching for the row index, but it works
            annotation_row = -1
            for j in range(len(annotations)):
                if annotations[j][0] == song_id:
                    annotation_row = j
                    break

            if not annotation_row == -1:
                row_to_write += ',' + annotations[annotation_row][1] + ',' + annotations[annotation_row][2]

            output.write(row_to_write + '\n')

#convert_mp3_to_wav(mp3_dir, wav_dir)
#extract_features_per_frame(wav_dir, csv_dir, 90)
#extract_features_per_song(wav_dir, csv_file)

#there should be two variants of feature extraction (in two separate scripts): one that uses frame size of 500 msec (sampling rate of 2 Hz) and feature extraction per song level - using entire song (45 sec) as a frame size
#print(combine_static_annotation(stat_ann_dir))
merge_features_annontations_per_song(csv_file, stat_ann_dir)

#in the first variant there will be one csv file per song with columns that correspond to features and rows to windows
#in the second script the output will be one csv file for features as columns and rows as songs
