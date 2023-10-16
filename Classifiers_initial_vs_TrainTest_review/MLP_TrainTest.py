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
from sklearn.neural_network import MLPClassifier
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


name = "MLP_TrainTest"
file0 = open("{}_parameters.data".format(name), "w")
file1 = open("{}_results.data".format(name), "w")




def MLPParRandomSearch():
    parameters = []
    def hidLayerSize():
        numHidLayers = random.randint(2,5)
        HLS = []
        for i in range(numHidLayers):
            HLS.append(random.randint(10,100))
        return tuple(HLS)
    #Choosing activation function
    ActFun = random.choice(['identity', 'logistic', 'tanh', 'relu'])
    #Choosing solver
    Solver = random.choice(['lbfgs', 'sgd', 'adam'])
    #Choosing Alpha parameter L2 penalty parameter
    Alpha = random.uniform(1e-6, 1e-2)
    #Choosing learning rate
    LearnRate = random.choice(['constant', 'invscaling', 'adaptive'])
    #Choosing maximum number of iterations
    MaxIter = random.randint(200,2000)
    #Choosing Tolerance
    Tol = random.uniform(1e-10, 1e-4)
    #Maximum number of iterations without change
    while True:
        nIter = random.randint(10,10000)
        if nIter < MaxIter:
            print("nIter smaller than MaxIter")
            break
        else:
            continue
    ######################################################
    #Appending Randomly Selected Data into Parameters list
    ######################################################
    parameters.append(hidLayerSize())
    parameters.append(ActFun)
    parameters.append(Solver)
    parameters.append(Alpha)
    parameters.append(LearnRate)
    parameters.append(MaxIter)
    parameters.append(Tol)
    parameters.append(nIter)
    #####################################################
    # Writing parameters into file0 _parameters.dat
    #####################################################
    file0.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(parameters[0],
                                                          parameters[1],
                                                          parameters[2],
                                                          parameters[3],
                                                          parameters[4],
                                                          parameters[5],
                                                          parameters[6],
                                                          parameters[7]))
    file0.flush()
    return parameters


def MLPReg(parameters, X_train,y_train,X_test, y_test):
    model = MLPClassifier(hidden_layer_sizes=parameters[0],
                          activation= parameters[1],
                          solver= parameters[2],
                          alpha= parameters[3],
                          learning_rate = parameters[4],
                          max_iter = parameters[5],
                          tol = parameters[6],
                          n_iter_no_change = parameters[7])
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
MLP_BestParams = []

while True:
    print("Current Iteration = {}".format(k))
    Param = MLPParRandomSearch()
    test = MLPReg(Param,X_train,y_train, X_test, y_test)
    k+=1
    if (test[8] > 0.999):
        BestScores = test
        print("Solution is Found!!")
        file1.write("Solution is Found!")
        file1.flush()
        break
    elif (test[8] > BestScores[8]):
        BestScores = test
        MLP_BestParams = Param
    elif (k >= 1000):
        print("Algorithm has reached maximum number of iterations!!\n")
        print("Best score = {}".format(BestScores[8]))
        print("Best parameters = {}".format(MLP_BestParams))
        file1.write("Algorithm has reached maximum number of iterations!!")
        file1.flush()
        break
    else:
        continue
file0.close()
file1.close()


MLP_BestMeanScores = [BestScores[0], BestScores[2], BestScores[4], BestScores[6], BestScores[8]]

MLP_BestStdScores = [BestScores[1], BestScores[3], BestScores[5], BestScores[7], BestScores[9]]


