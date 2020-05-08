'''
Created on May 7, 2020
@author: kaytee
Tests different regression models for valence and arousal
'''
from pyAudioAnalysis import audioTrainTest
import numpy as np

#Read features and labels from csv files
csv_song = "dataset//extracted_features//features_no_sampling_avg_merge.csv"
csv_frame = "dataset//extracted_features//features_sampling_merged.csv"

def read_features_csv(file):
    return np.genfromtxt(file,delimiter=',',skip_header=1,usecols = range(1,35))
    
    
def read_valence(file):
    return np.genfromtxt(file,delimiter=',',skip_header=1,usecols = (36))
    
def read_arousal(file):
    return np.genfromtxt(file,delimiter=',',skip_header=1,usecols = (35))

#TO DO Normalization of features?


# Tuning parameters
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
c_param_4 = 1000
c_param_5 = 10000
c_param_6 = 100000
c_param_7 = 1000000

#variant for random forest - nr of estomators
n_1 = 10
n_2 = 100

#build and evaluate regression

'''
Perform training and evaluation of different regression models on song-level dataset
'''
def train_models_song(csv_song):
    
    #read features from file
    features_per_song = read_features_csv(csv_song)
    #print(features_per_song)
    
    #read labels from file 
    
    valence_labels_song = read_valence(csv_song)
    #print(valence_labels_song)
    
    arousal_labels_song = read_arousal(csv_song)
    #print(arousal_labels_song)
    
    #optional normalization
    features_norm, mean, std = audioTrainTest.normalize_features(features_per_song)
    
    #Prints RMSE root mean squared error
    #svm_model, error = audioTrainTest.train_svm_regression(features_per_song, arousal_labels_song, c_param_1, kernel='linear')
    #print(error)
    
    #svm_model, error = audioTrainTest.train_svm_regression(features_per_song, arousal_labels_song, c_param_2, kernel='linear')
    #print(error)
    
    #svm_model, error = audioTrainTest.train_svm_regression(features_per_song, arousal_labels_song, c_param_3, kernel='linear')
    #print(error)
    
    #svm_model, error = audioTrainTest.train_svm_regression(features_per_song, arousal_labels_song, c_param_6, kernel='poly')
    #print(error)
    
    rf_model, error = audioTrainTest.train_random_forest_regression(features_per_song, valence_labels_song, 50)
    print(error)
    

'''
PER SONG DATASET
*****************Tuning***************************************
SVM
Kernel    C        RMSE_arousal            RMSE_valence
linear     c1      0.7826529105769882
linear     c2      0.769494156112072
linear     c3      0.7541538268819094    0.6869591733288045 <-----OPT_LIN
rbf        c3      0.7649733193215299
rbf        c6      0.6643038703578852    0.6540281566625666 <----OPT_RBF
rbf        c5                             0.6869591733288045
sigmoid    0.001    1.070402737059893
poly        c3      0.7744015664233957
poly        c4      0.7489752120251861
poly        c6      0.6894546907917157

Random forest
#Est                
50                0.29432147239263795   0.27406723926380366 <-----OPT_RF
60                0.2888088957055214 <- 0.2765730061349693 
70                0.28907467134092896
40                0.29143358895705523    0.2787717791411043
'''

'''
Perform training and evaluation of different regression models on frame-level dataset
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
    
    #optional normalization
    features_norm, mean, std = audioTrainTest.normalize_features(features_per_frame)
    
    svm_model, error = audioTrainTest.train_svm_regression(features_per_frame, arousal_labels_frame, c_param_3, kernel='linear')
    print(error)
    
    #rf_model, error = audioTrainTest.train_random_forest_regression(features_per_frame, valence_labels_frame, 50)
    #print(error)

#results
'''
PER FRAME DATASET
*****************Tuning***************************************
Random forest    Arousal-RMSE                Valence-RMSE
#Est                
50                                         0.052537364826242265<-----OPT_RF
60                0.05273607016219178 <- 
70                
40  



SVM
Kernel    C        RMSE_arousal            RMSE_valence
linear     c1      0.7826529105769882
linear     c2      0.769494156112072
linear     c3      0.7541538268819094    0.6869591733288045 <-----OPT_LIN
rbf        c3      0.7649733193215299
rbf        c6      0.6643038703578852    0.6540281566625666 <----OPT_RBF
rbf        c5                             0.6869591733288045
sigmoid    0.001    1.070402737059893
poly        c3      0.7744015664233957
poly        c4      0.7489752120251861
poly        c6      0.6894546907917157

Note:              
'''


#parameter optimization TO DO
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
    
    
#train_models_song(csv_song)

train_models_frame(csv_frame)