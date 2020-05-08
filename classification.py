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

#variant for random forest - nr of estimators
n_1 = 10
n_2 = 100

#build and evaluate regression
'''
ARGUMENTS:
        dataset:    "song" or "frame"
        model:   "svm" or "randomforest"
        kernel: kernel of SVM model: 'rbf', 'linear', 'poly', 'sigmoid'
        c_param: C parameter for RBF
        est:    nr of estimators for random forests [10;100]
        label:    "valence" or "arousal"
    RETURNS:
        error:     root mean squared error
'''
def train_model(dataset="song", normalize = False, model = "svm", kernel='rbf', c_param=10, label="valence", est=50):
    
    if dataset == "song":
        file = csv_song
    elif dataset == "frame":
        file = csv_frame
    
    features = read_features_csv(file)
    
    if normalize:
        features = audioTrainTest.normalize_features(features)
    if label == "valence":
        labels = read_valence(file)
    elif label == "arousal":
        labels = read_arousal(file)
    print("Nr of instances: "+ str(len(labels)))
    
    if model == "svm":
        reg_model, error = audioTrainTest.train_svm_regression(features, labels, c_param, kernel)
    elif model == "randomforest":
        reg_model, error = audioTrainTest.train_random_forest_regression(features, labels, est)
        
    print("Error: " + str(error))
    
    return reg_model, error

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

            
'''

"""
 Checks regression model with different paramaters and returns the optimal
    ARGUMENTS:
        file:    file with features and lables
        n_exp:         number of cross-validation experiments
        method_name:   "svm" or "randomforest"
        label:    "valence" or "arousal"
    RETURNS:
         bestParam:   the value of the input parameter that optimizes
         the selected performance measure
"""
def evaluate(file, model_type='randomforest', n_exp=10, label='valence'):

    features = read_features_csv(file)
    if label == "valence":
        labels = read_valence(file)
    elif label == "arousal":
        labels = read_arousal(file)
    
    if model_type == "svm" or model_type == "svm_rbf":
        model_params = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5,
                                 1.0, 5.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])
    elif model_type == "randomforest":
        model_params = np.array([5, 10, 25, 50, 100])
    
    print(model_params)

    bestParam, error, berror = audioTrainTest.evaluate_regression(features, labels, n_exp, model_type, model_params)
    print("Best param:")
    print(bestParam)
    print("Error:")
    print(error)
    print("Best error:")
    print (berror)
#bestParam = evaluate_regression(features, labels, n_exp, method_name, params):
    
#train_models_song(csv_song)

#train_models_frame(csv_frame)

#evaluate(file=csv_song, n_exp=10)

train_model(dataset="song", normalize = False, model = "svm", kernel='rbf', c_param=10, label="valence", est=50)