'''
Created on May 7, 2020

@author: kaytee

Tests different regression models for valence and arousal
'''
from pyAudioAnalysis import audioTrainTest
import numpy as np


csv_song = "dataset//extracted_features//features_no_sampling_avg_merge.csv"
csv_frame = "dataset//extracted_features//features_sampling_merged.csv"

def read_features_csv(file):
    return np.genfromtxt(file,delimiter=',',skip_header=1,usecols = range(1,35))
    
    
def read_valence(file):
    return np.genfromtxt(file,delimiter=',',skip_header=1,usecols = (36))
    
def read_arousal(file):
    return np.genfromtxt(file,delimiter=',',skip_header=1,usecols = (35))

n_exp=10;
method1="svm"
method2="randomforest"


#variants for SVM kernel
kernel_1 = 'linear'
kernel_2 = 'poly'
kernel_3 = 'sigmoid'
kernel_4 = 'rbf'

# variants for c in SVM RBF
# see https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
c_param_1 = 0.1
c_param_2 = 1
c_param_3 = 100


def train_models_song(csv_song):
    
    #read features from file
    features_per_song = read_features_csv(csv_song)
    #print(features_per_song)
    
    #read labels from file 
    
    valence_labels_song = read_valence(csv_song)
    #print(valence_labels_song)
    
    arousal_labels_song = read_arousal(csv_song)
    #print(arousal_labels_song)
    
    #Prints RMSE root mean squared error
    #svm_model, error = audioTrainTest.train_svm_regression(features_per_song, arousal_labels_song, c_param_1, kernel='linear')
    #print(error)
    
    #svm_model, error = audioTrainTest.train_svm_regression(features_per_song, arousal_labels_song, c_param_2, kernel='linear')
    #print(error)
    
    #svm_model, error = audioTrainTest.train_svm_regression(features_per_song, arousal_labels_song, c_param_3, kernel='linear')
    #print(error)
    
    svm_model, error = audioTrainTest.train_svm_regression(features_per_song, arousal_labels_song, c_param_3, kernel='rbf')
    print(error)
    
    

'''
linear c1 0.7826529105769882
linear c2 0.769494156112072
linear c3 0.7541538268819094
rbf    c3    0.7649733193215299


'''

def train_models_frame(file):
    #read features from file
    features_per_frame = read_features_csv(csv_frame)
    #print(features_per_frame)
    #read labels from file 
    
    valence_labels_frame = read_valence(csv_frame)
    #print(valence_labels_frame)
    
    arousal_labels_frame = read_arousal(csv_frame)
    #print(arousal_labels_frame)

#build and evaluate regression

train_models_song(csv_song)

#TODO


#

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

