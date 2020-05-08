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

#def train_models(dataset="song", normalize = False, model = "svm", kernel='rbf', c_param=10, label="valence")

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
Kernel    C        Train_err_arousal            Train_err_RMSE_valence
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
Random forest    Arousal-train_err                Valence-train_err
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
        svm_kernel:    'linear', 'rbf', 'poly', 'sigmoid'
        label:    "valence" or "arousal"
    RETURNS:
         bestParam:   the value of the input parameter that optimizes
         the selected performance measure
"""
def evaluate(file, model_type='randomforest', svm_kernel = 'linear', n_exp=10, label='valence'):

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

    #bestParam, error, berror = audioTrainTest.evaluate_regression(features, labels, n_exp, model_type, model_params)
    bestParam, error, berror = evaluate_regression(features, labels, n_exp, model_type, svm_kernel, model_params, normalize=False, per_train=0.9)
    print("Testing model: "+ model_type)
    print("Best param:")
    print(bestParam)
    print("Error:")
    print(error)
    print("Best error:")
    print (berror)

"Override from audioTrainTest - with no normalization option, percentage of train, and different kernels"
def evaluate_regression(features, labels, n_exp, method_name, svm_kernel, params, normalize=False, per_train=0.9):
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

    # feature normalization:
    if (normalize == True):
        features_norm, mean, std = audioTrainTest.normalize_features([features])
        features_norm = features_norm[0]
    elif (normalize == False):
        features_norm=features
    n_samples = labels.shape[0]
    #per_train = 0.9
    errors_all = []
    er_train_all = []
    er_base_all = []
    for Ci, C in enumerate(params):   # for each param value
                errors = []
                errors_train = []
                errors_baseline = []
                for e in range(n_exp):   # for each cross-validation iteration:
                    # split features:
                    randperm = np.random.permutation(range(n_samples))
                    n_train = int(round(per_train * n_samples))
                    f_train = [features_norm[randperm[i]]
                               for i in range(n_train)]
                    f_test = [features_norm[randperm[i+n_train]]
                              for i in range(n_samples - n_train)]
                    l_train = [labels[randperm[i]] for i in range(n_train)]
                    l_test = [labels[randperm[i + n_train]]
                              for i in range(n_samples - n_train)]

                    # train multi-class svms:                    
                    f_train = np.matrix(f_train)                                 
                    if method_name == "svm":                                        
                        classifier, train_err = \
                            audioTrainTest.train_svm_regression(f_train, l_train, C, kernel=svm_kernel)
                    #elif method_name == "svm_rbf":                      
                        #classifier, train_err = \
                            #train_svm_regression(f_train, l_train, C,
                                                 #kernel='rbf')
                    elif method_name == "randomforest":
                        classifier, train_err = \
                            audioTrainTest.train_random_forest_regression(f_train, l_train, C)
                    error_test = []
                    error_test_baseline = []
                    for itest, fTest in enumerate(f_test):
                        R = audioTrainTest.regression_wrapper(classifier, method_name, fTest)
                        Rbaseline = np.mean(l_train)
                        error_test.append((R - l_test[itest]) *
                                          (R - l_test[itest]))
                        error_test_baseline.append((Rbaseline - l_test[itest]) *
                                                  (Rbaseline - l_test[itest]))
                    error = np.array(error_test).mean()
                    error_baseline = np.array(error_test_baseline).mean()
                    errors.append(error)
                    errors_train.append(train_err)
                    errors_baseline.append(error_baseline)
                errors_all.append(np.array(errors).mean())
                er_train_all.append(np.array(errors_train).mean())
                er_base_all.append(np.array(errors_baseline).mean())

    best_ind = np.argmin(errors_all)

    print("{0:s}\t\t{1:s}\t\t{2:s}\t\t{3:s}".format("Param", "MSE",
                                                    "T-MSE", "R-MSE"))
    for i in range(len(errors_all)):
        print("{0:.4f}\t\t{1:.2f}\t\t{2:.2f}\t\t{3:.2f}".format(params[i],
                                                                errors_all[i],
                                                                er_train_all[i],
                                                                er_base_all[i]),
              end="")
        if i == best_ind:
            print("\t\t best",end="")
        print("")
    return params[best_ind], errors_all[best_ind], er_base_all[best_ind]
#bestParam = evaluate_regression(features, labels, n_exp, method_name, params):
    
#train_models_song(csv_song)

#train_models_frame(csv_frame)

#evaluate(file=csv_song, n_exp=10)
evaluate(file=csv_song, model_type='svm', svm_kernel = 'linear', n_exp=10, label='valence')

#train_model(dataset="song", normalize = False, model = "svm", kernel='rbf', c_param=10, label="valence", est=50)