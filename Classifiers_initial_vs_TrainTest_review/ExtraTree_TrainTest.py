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
from sklearn.tree import ExtraTreeClassifier
import random


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


name = "ExtraTree_TrainTest"
file0 = open("{}_parameters.data".format(name), "w")
file1 = open("{}_results.data".format(name), "w")


def ExtraTreeParRandomSearch():
    parameters = []
    #Function to measure the quality of a split
    Criterion = random.choice(['gini', 'entropy'])
    #Maximum depth of the tree
    MaxDepth = random.randint(10,20)
    #The minimum number of samples required to be at a leaf node
    MinSamplLeaf = random.randint(1,10)
    MaxLeafNodes = random.choice([None, 60,70,80,90,100])
    ######################################################
    #Appending Randomly Selected Data into Parameters list
    ######################################################
    parameters.append(Criterion)
    parameters.append(MaxDepth)
    parameters.append(MinSamplLeaf)
    parameters.append(MaxLeafNodes)
    #####################################################
    # Writing parameters into file0 _parameters.dat
    #####################################################
    file0.write("{}\t{}\t{}\t{}\n".format(parameters[0],
                                              parameters[1],
                                              parameters[2],
                                              parameters[3]))
    file0.flush()
    return parameters


def ExtraTreeReg(parameters, X_train,y_train,X_test, y_test):
    model = ExtraTreeClassifier(criterion= parameters[0],
                                   max_depth= parameters[1],
                                   min_samples_leaf = parameters[2],
                                   max_leaf_nodes= parameters[3])
    model.fit(X_train, y_train)
    print("###################################################################")
    ##########################################################################
    #Train Test Scores Raw####################################################
    ##########################################################################
    print("###################################################################")
    print("# Calculate Mean and Standard Deviation of Metric values ")
    print("###################################################################")
    ACCScoreTrain = model.score(X_train, y_train)
    ACCScoreTest = model.score(X_test, y_test)
    AvrACCScore = np.mean([ACCScoreTrain, ACCScoreTest])
    StdACCScore = np.std([ACCScoreTrain, ACCScoreTest])
    
    PRECISIONScoreTrain = precision_score(y_train, model.predict(X_train), average='weighted')
    PRECISIONScoreTest = precision_score(y_test, model.predict(X_test), average='weighted')
    AvrPRECISIONScore = np.mean([PRECISIONScoreTrain, PRECISIONScoreTest])
    StdPRECISIONScore = np.std([PRECISIONScoreTrain, PRECISIONScoreTest])
    
    RECALLScoreTrain = recall_score(y_train, model.predict(X_train), average='weighted')
    RECALLScoreTest = recall_score(y_test, model.predict(X_test), average='weighted')
    AvrRECALLScore = np.mean([RECALLScoreTrain, RECALLScoreTest])
    StdRECALLScore = np.std([RECALLScoreTrain, RECALLScoreTest])
    
    F1ScoreTrain = f1_score(y_train, model.predict(X_train), average='weighted')
    F1ScoreTest = f1_score(y_test, model.predict(X_test), average='weighted')
    AvrF1Score = np.mean([F1ScoreTrain, F1ScoreTest])
    StdF1Score = np.std([F1ScoreTrain, F1ScoreTest])
    
    ROCAUCScoreTrain = roc_auc_score(y_train, model.predict_proba(X_train), average='weighted', multi_class='ovr')
    ROCAUCScoreTest = roc_auc_score(y_test, model.predict_proba(X_test), average='weighted', multi_class='ovr')
    AvrROCAUCScore = np.mean([ROCAUCScoreTrain, ROCAUCScoreTest])
    StdROCAUCScore = np.std([ROCAUCScoreTrain, ROCAUCScoreTest])
    
    scores = []
    scores.append(AvrACCScore)
    scores.append(StdACCScore)
    scores.append(AvrPRECISIONScore)
    scores.append(StdPRECISIONScore)
    scores.append(AvrRECALLScore)
    scores.append(StdRECALLScore)
    scores.append(AvrF1Score)
    scores.append(StdF1Score)
    scores.append(AvrROCAUCScore)
    scores.append(StdROCAUCScore)
    
    print("ROC AUC Score = {}".format(AvrROCAUCScore))
    print("STD ROC AUC Score = {}".format(StdROCAUCScore))
    file1.write("##############################################################\n"+\
                "ACCScore Train = {}\n".format(ACCScoreTrain)+\
                "ACCScore Test = {}\n".format(ACCScoreTest)+\
                "AvrACCScore = {}\n".format(AvrACCScore)+\
                "StdACCScore = {}\n".format(StdACCScore)+\
                "PRECISIONScore Train = {}\n".format(PRECISIONScoreTrain)+\
                "PRECISIONScore Test = {}\n".format(PRECISIONScoreTest)+\
                "AvrPRECISIONScore = {}\n".format(AvrPRECISIONScore)+\
                "StdPRECISIONScore = {}\n".format(StdPRECISIONScore)+\
                "RECALLScore Train = {}\n".format(RECALLScoreTrain)+\
                "RECALLScore Test = {}\n".format(RECALLScoreTest)+\
                "AvrRECALLScore = {}\n".format(AvrRECALLScore)+\
                "StdRECALLScore = {}\n".format(StdRECALLScore)+\
                "F1Score Train = {}\n".format(F1ScoreTrain)+\
                "F1Score Test = {}\n".format(F1ScoreTest)+\
                "AvrF1Score = {}\n".format(AvrF1Score)+\
                "StdF1Score = {}\n".format(StdF1Score)+\
                "ROCAUCScore Train = {}\n".format(ROCAUCScoreTrain)+\
                "ROCAUCScore Test = {}\n".format(ROCAUCScoreTest)+\
                "AvrROCAUCScore = {}\n".format(AvrROCAUCScore)+\
                "StdROCAUCScore = {}\n".format(StdROCAUCScore)+\
                "###############################################################\n")
    file1.flush()
    return scores


k = 0
BestScores = [0]*10
ExtraTree_BestParams = []

while True:
    print("Current Iteration = {}".format(k))
    Param = ExtraTreeParRandomSearch()
    test = ExtraTreeReg(Param,X_train,y_train, X_test, y_test)
    k+=1
    if (test[8] > 0.999):
        BestScores = test
        print("Solution is Found!!")
        file1.write("Solution is Found!")
        file1.flush()
        break
    elif (test[8] > BestScores[8]):
        BestScores = test
        ExtraTree_BestParams = Param
    elif (k >= 1000):
        print("Algorithm has reached maximum number of iterations!!\n")
        print("Best score = {}".format(BestScores[8]))
        print("Best parameters = {}".format(ExtraTree_BestParams))
        file1.write("Algorithm has reached maximum number of iterations!!")
        file1.flush()
        break
    else:
        continue
file0.close()
file1.close()



ExtraTree_BestMeanScores = [BestScores[0], BestScores[2], BestScores[4], BestScores[6], BestScores[8]]

ExtraTree_BestStdScores = [BestScores[1], BestScores[3], BestScores[5], BestScores[7], BestScores[9]]


