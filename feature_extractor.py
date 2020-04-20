'''
Created on Apr 16, 2020

@author: kaytee


'''
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioAnalysis
from pyAudioAnalysis import ShortTermFeatures
from pydub import AudioSegment

#if there is error with audio library path to ffmpeg needs to be explicitly defined
#AudioSegment.converter = r"C:\\Users\\kaytee\\Documents\\Installation Programs\\ffmpeg-20200417-889ad93-win64-static\\ffmpeg-20200417-889ad93-win64-static\\bin\\ffmpeg.exe"

# 1st step - convert files from mp3 to wav, example for one song below
src = "2.mp3"
dst = "2.wav"

# convert mp3 to wav

sound = AudioSegment.from_mp3(src)
sound.export(dst, format = "wav")

#perform audio feature extraction using ShortTermFeatures: output: csv file with song id and numeric values for 34 audio features

