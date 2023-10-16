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

model = MLPClassifier(random_state=1).fit(X_train, y_train)

AccTrain = model.score(X_train, y_train)
AccTest = model.score(X_test, y_test)
MLP_AccMean = np.mean([AccTrain, AccTest]) 
MLP_AccStd = np.std([AccTrain, AccTest])

PrecisionTrain = precision_score(y_train, model.predict(X_train), average='weighted')
PrecisionTest = precision_score(y_test, model.predict(X_test), average='weighted')
MLP_PrecisionMean = np.mean([PrecisionTrain, PrecisionTest]) 
MLP_PrecisionStd = np.std([PrecisionTrain, PrecisionTest])

RecallTrain = recall_score(y_train, model.predict(X_train), average='weighted')
RecallTest = recall_score(y_test, model.predict(X_test), average='weighted')
MLP_RecallMean = np.mean([RecallTrain, RecallTest]) 
MLP_RecallStd = np.std([RecallTrain, RecallTest])

F1Train = f1_score(y_train, model.predict(X_train), average='weighted')
F1Test = f1_score(y_test, model.predict(X_test), average='weighted')
MLP_F1Mean = np.mean([F1Train, F1Test]) 
MLP_F1Std = np.std([F1Train, F1Test])

RocAucTrain = roc_auc_score(y_train, model.predict_proba(X_train), average='weighted', multi_class='ovr')
RocAucTest = roc_auc_score(y_test, model.predict_proba(X_test), average='weighted', multi_class='ovr')
MLP_RocAucMean = np.mean([RocAucTrain, RocAucTest]) 
MLP_RocAucStd = np.std([RocAucTrain, RocAucTest])

MLP_MeanScores = [MLP_AccMean, MLP_PrecisionMean, MLP_RecallMean, 
                  MLP_F1Mean, MLP_RocAucMean]

MLP_StdScores = [MLP_AccStd, MLP_PrecisionStd, MLP_RecallStd,
                 MLP_F1Std, MLP_RocAucStd]

print("########################################################################")
print("# MLP Evaluation values ################################################")
print("########################################################################")
print("ACC mean score = {}".format(MLP_AccMean))
print("ACC std  = {}".format(MLP_AccStd))
print("########################################################################")
print("PRECISION mean score = {}".format(MLP_PrecisionMean))
print("PRECISION std  = {}".format(MLP_PrecisionStd))
print("########################################################################")
print("RECALL mean score = {}".format(MLP_RecallMean))
print("RECALL std  = {}".format(MLP_RecallStd))
print("########################################################################")
print("F1 mean score = {}".format(MLP_F1Mean))
print("F1 std  = {}".format(MLP_F1Std))
print("########################################################################")
print("ROC AUC mean score = {}".format(MLP_RocAucMean))
print("ROC AUC std  = {}".format(MLP_RocAucStd))



