"""
@author: Luka Ilijanic
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.ensemble import StackingClassifier


file_name2 = "C:/Users/Luka/Desktop/archive/car_evaluation_coded.csv" 
data2 = pd.read_csv(file_name2)


X = pd.DataFrame()
X['buying_price'] = data2['buying_price']
X['maint_cost'] = data2['maint_cost']
#X['no_doors'] = data2['no_doors']
X['no_persons'] = data2['no_persons']
X['lug_boot'] = data2 ['lug_boot']
X['safety'] = data2['safety']
y = data2['decision']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)


name = "StackEnsembCV10"
file1 = open("{}_CV10_score.dat".format(name),'w')


def StackingEnsemble(X_train,X_test, y_train, y_test):
    estimators = [('KNN', KNeighborsClassifier(n_neighbors = 11,
                                               weights = 'uniform',
                                               algorithm = 'brute')),
                  ('RadiusN', RadiusNeighborsClassifier(radius = 1.0265985133064193,
                                                        weights = 'uniform',
                                                        algorithm = 'auto')),
                    ('MLP', MLPClassifier(hidden_layer_sizes=(82, 31),
                                          activation= 'logistic',
                                          solver= 'lbfgs',
                                          alpha= 0.006827331832579011,
                                          learning_rate = 'adaptive',
                                          max_iter = 1375,
                                          tol = 6.732278092909502e-05,
                                          n_iter_no_change = 1272)),
                  ('DecTree', DecisionTreeClassifier(criterion= 'gini',
                                                     splitter= 'random',
                                                     max_depth= 11,
                                                     min_samples_leaf = 8,
                                                     max_leaf_nodes= None)),
                  ('ExtraTree', ExtraTreeClassifier(criterion= 'entropy',
                                                    max_depth= 19,
                                                    min_samples_leaf = 4,
                                                    max_leaf_nodes= 100))]
      
    model = StackingClassifier(estimators=estimators, 
                               final_estimator = None)
    cvmodel = cross_validate(model, X_train, y_train, cv=10, 
                              scoring = ("roc_auc_ovr_weighted",
                                         "precision_weighted", 
                                         "recall_weighted",
                                         "f1_weighted"),
                              return_train_score=True)
    print("###################################################################")
    print("# Results from CV 10 Cross Validation Using Multiple Metric")
    print("###################################################################")
    print("All Scores From CV 10 = {}".format(cvmodel))
    file1.write("###############################################################\n")
    file1.write(" Results from CV 10\n")
    file1.write("################################################################\n")
    file1.write("ROC AUC Train Scores = {}\n".format(cvmodel['train_roc_auc_ovr_weighted']))
    file1.write("ROC AUC Test Scores = {}\n".format(cvmodel['test_roc_auc_ovr_weighted']))
    file1.write("PRECISION Train Scores = {}\n".format(cvmodel['train_precision_weighted']))
    file1.write("PRECISION Test Scores = {}\n".format(cvmodel['test_precision_weighted']))
    file1.write("RECALL Train Scores = {}\n".format(cvmodel['train_recall_weighted']))
    file1.write("RECALL Test Scores = {}\n".format(cvmodel['test_recall_weighted']))
    file1.write("F1 Train Scores = {}\n".format(cvmodel['train_f1_weighted']))
    file1.write("F1 Test Scores = {}\n".format(cvmodel['test_f1_weighted']))
    print("###################################################################")
    print("# Calculate Mean and Standard Deviation of Metric values ")
    print("###################################################################")
    AvrROCAUCScoreTrain = np.mean(cvmodel['train_roc_auc_ovr_weighted'])
    StdROCAUCScoreTrain = np.std(cvmodel['train_roc_auc_ovr_weighted'])
    AvrROCAUCScoreTest = np.mean(cvmodel['test_roc_auc_ovr_weighted'])
    StdROCAUCScoreTest = np.std(cvmodel['test_roc_auc_ovr_weighted'])
    AvrAllROCAUCScore = np.mean([AvrROCAUCScoreTrain, AvrROCAUCScoreTest])
    StdAllROCAUCScore = np.std([AvrROCAUCScoreTrain, AvrROCAUCScoreTest])
    
    AvrPRECISIONScoreTrain = np.mean(cvmodel['train_precision_weighted'])
    StdPRECISIONScoreTrain = np.std(cvmodel['train_precision_weighted'])
    AvrPRECISIONScoreTest = np.mean(cvmodel['test_precision_weighted'])
    StdPRECISIONScoreTest = np.std(cvmodel['test_precision_weighted'])
    AvrAllPRECISIONScore = np.mean([AvrPRECISIONScoreTrain, AvrPRECISIONScoreTest])
    StdAllPRECISIONScore = np.std([AvrPRECISIONScoreTrain, AvrPRECISIONScoreTest])
    
    AvrRECALLScoreTrain = np.mean(cvmodel['train_recall_weighted'])
    StdRECALLScoreTrain = np.std(cvmodel['train_recall_weighted'])
    AvrRECALLScoreTest = np.mean(cvmodel['test_recall_weighted'])
    StdRECALLScoreTest = np.std(cvmodel['test_recall_weighted'])
    AvrAllRECALLScore = np.mean([AvrRECALLScoreTrain, AvrRECALLScoreTest])
    StdAllRECALLScore = np.std([AvrRECALLScoreTrain, AvrRECALLScoreTest])
    
    AvrF1ScoreTrain = np.mean(cvmodel['train_f1_weighted'])
    StdF1ScoreTrain = np.std(cvmodel['train_f1_weighted'])
    AvrF1ScoreTest = np.mean(cvmodel['test_f1_weighted'])
    StdF1ScoreTest = np.std(cvmodel['test_f1_weighted'])
    AvrAllF1Score = np.mean([AvrF1ScoreTrain, AvrF1ScoreTest])
    StdAllF1Score = np.std([AvrF1ScoreTrain, AvrF1ScoreTest])
    
    print("CV-ROC AUC Score = {}".format(AvrAllROCAUCScore))
    print("CV-STD ROC AUC Score = {}".format(StdAllROCAUCScore))
    print("CV-PRECISION Score = {}".format(AvrAllPRECISIONScore))
    print("CV-STD PRECISION Score = {}".format(StdAllPRECISIONScore))
    print("CV-RECALL Score = {}".format(AvrAllRECALLScore))
    print("CV-STD RECALL Score = {}".format(StdAllRECALLScore))
    print("CV-F1 Score = {}".format(AvrAllF1Score))
    print("CV-STD F1 Score = {}".format(StdAllF1Score))
    file1.write("##############################################################\n"+\
                "AvrROCAUCScore Train = {}\n".format(AvrROCAUCScoreTrain)+\
                "StdROCAUCScore Train = {}\n".format(StdROCAUCScoreTrain)+\
                "AvrROCAUCScore Test = {}\n".format(AvrROCAUCScoreTest)+\
                "StdROCAUCScore Test = {}\n".format(StdROCAUCScoreTest)+\
                "AvrAllROCAUCScore = {}\n".format(AvrAllROCAUCScore)+\
                "StdAllROCAUCScore = {}\n".format(StdAllROCAUCScore)+\
                "AvrPRECISIONScore Train = {}\n".format(AvrPRECISIONScoreTrain)+\
                "StdPRECISIONScore Train = {}\n".format(StdPRECISIONScoreTrain)+\
                "AvrPRECISIONScore Test = {}\n".format(AvrPRECISIONScoreTest)+\
                "StdPRECISIONScore Test = {}\n".format(StdPRECISIONScoreTest)+\
                "AvrAllPRECISIONScore = {}\n".format(AvrAllPRECISIONScore)+\
                "StdAllPRECISIONScore = {}\n".format(StdAllPRECISIONScore)+\
                "AvrRECALLScore Train = {}\n".format(AvrRECALLScoreTrain)+\
                "StdRECALLScore Train = {}\n".format(StdRECALLScoreTrain)+\
                "AvrRECALLScore Test = {}\n".format(AvrRECALLScoreTest)+\
                "StdRECALLScore Test = {}\n".format(StdRECALLScoreTest)+\
                "AvrAllRECALLScore = {}\n".format(AvrAllRECALLScore)+\
                "StdAllRECALLScore = {}\n".format(StdAllRECALLScore)+\
                "AvrF1Score Train = {}\n".format(AvrF1ScoreTrain)+\
                "StdF1Score Train = {}\n".format(StdF1ScoreTrain)+\
                "AvrF1Score Test = {}\n".format(AvrF1ScoreTest)+\
                "StdF1Score Test = {}\n".format(StdF1ScoreTest)+\
                "AvrAllF1Score = {}\n".format(AvrAllF1Score)+\
                "StdAllF1Score = {}\n".format(StdAllF1Score)+\
                "###############################################################\n")
    print("###############################################################")
    print(" Train/Test scores")
    print("################################################################")
    file1.write("###############################################################\n")
    file1.write(" Train/Test scores\n")
    file1.write("################################################################\n")
    model.fit(X_train,y_train)
    ROCAUCTrain = roc_auc_score(y_train, model.predict_proba(X_train), average='weighted', multi_class='ovr')
    ROCAUCTest = roc_auc_score(y_test, model.predict_proba(X_test), average='weighted', multi_class='ovr')
    ROCAUCMean = np.mean([ROCAUCTrain, ROCAUCTest])
    PRECISIONTrain = precision_score(y_train, model.predict(X_train), average='weighted')
    PRECISIONTest = precision_score(y_test, model.predict(X_test), average='weighted')
    PRECISIONMean = np.mean([PRECISIONTrain, PRECISIONTest])
    RECALLTrain = recall_score(y_train, model.predict(X_train), average='weighted')
    RECALLTest = recall_score(y_test, model.predict(X_test), average='weighted')
    RECALLMean = np.mean([RECALLTrain, RECALLTest])
    F1Train = f1_score(y_train, model.predict(X_train), average='weighted')
    F1Test = f1_score(y_test, model.predict(X_test), average='weighted')
    F1Mean = np.mean([F1Train, F1Test])
    print("ROC AUC score = {}".format(ROCAUCMean))
    print("PRECISION score = {}".format(PRECISIONMean))
    print("RECALL score = {}".format(RECALLMean))
    print("F1 score = {}".format(F1Mean))
    file1.write("###############################################################\n")
    file1.write("ROC AUC score = {}\n".format(ROCAUCMean))
    file1.write("PRECISION score = {}\n".format(PRECISIONMean))
    file1.write("RECALL score = {}\n".format(RECALLMean))
    file1.write("F1 score = {}\n".format(F1Mean))
    file1.write("###############################################################\n")
    file1.flush()
    return ROCAUCMean


StackingEnsemble(X_train, X_test, y_train, y_test)


