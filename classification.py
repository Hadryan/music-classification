'''
Created on May 7, 2020

@author: kaytee

Tests different regression models for valence and arousal
'''
from pyAudioAnalysis import audioTrainTest
import numpy as np


csv_song = "dataset//extracted_features//features_no_sampling_merge.csv"
csv_frame = "dataset//extracted_features//features_sampling_merged.csv"

def read_features_csv(file):
    np.genfromtxt(file,delimiter=',',skiprows=1,usecols = range(1,35))
    
def read_valence(file):
    np.genfromtxt(file,delimiter=',',skiprows=1,usecols = (36))
    
def read_arousal(file):
    np.genfromtxt(file,delimiter=',',skiprows=1,usecols = (35))

#read features from file
features_per_song = read_features_csv(csv_song)

#read labels from file 

valence_labels_song = read_valence(csv_song)

arousal_labels_song = read_arousal(csv_song)


#read features from file
features_per_frame = read_features_csv(csv_frame)

#read labels from file 

valence_labels_frame = read_valence(csv_frame)

arousal_labels_frame = read_arousal(csv_frame)

#build and evaluate regression

n_exp=10;
method1="svm"
method2="randomforest"

#TODO
#svm_model, error = train_svm_regression(features, labels, c_param, kernel='linear')


#rf_model, error = train_random_forest_regression(features, labels, n_estimators)


#bestParam = evaluate_regression(features, labels, n_exp, method_name, params):
"""
    ARGUMENTS:
        features:     np matrices of features [n_samples x numOfDimensions]
        labels:       list of sample labels
        n_exp:         number of cross-validation experiments
        method_name:   "svm" or "randomforest"
        params:       list of classifier params to be evaluated
    RETURNS:
         bestParam:   the value of the input parameter that optimizes
         the selected performance measure
    """

