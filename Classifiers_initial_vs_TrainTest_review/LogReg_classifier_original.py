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
from sklearn.linear_model import LogisticRegression


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

model = LogisticRegression().fit(X_train, y_train)

AccTrain = model.score(X_train, y_train)
AccTest = model.score(X_test, y_test)
LogReg_AccMean = np.mean([AccTrain, AccTest]) 
LogReg_AccStd = np.std([AccTrain, AccTest])

PrecisionTrain = precision_score(y_train, model.predict(X_train), average='weighted')
PrecisionTest = precision_score(y_test, model.predict(X_test), average='weighted')
LogReg_PrecisionMean = np.mean([PrecisionTrain, PrecisionTest]) 
LogReg_PrecisionStd = np.std([PrecisionTrain, PrecisionTest])

RecallTrain = recall_score(y_train, model.predict(X_train), average='weighted')
RecallTest = recall_score(y_test, model.predict(X_test), average='weighted')
LogReg_RecallMean = np.mean([RecallTrain, RecallTest]) 
LogReg_RecallStd = np.std([RecallTrain, RecallTest])

F1Train = f1_score(y_train, model.predict(X_train), average='weighted')
F1Test = f1_score(y_test, model.predict(X_test), average='weighted')
LogReg_F1Mean = np.mean([F1Train, F1Test]) 
LogReg_F1Std = np.std([F1Train, F1Test])

RocAucTrain = roc_auc_score(y_train, model.predict_proba(X_train), average='weighted', multi_class='ovr')
RocAucTest = roc_auc_score(y_test, model.predict_proba(X_test), average='weighted', multi_class='ovr')
LogReg_RocAucMean = np.mean([RocAucTrain, RocAucTest]) 
LogReg_RocAucStd = np.std([RocAucTrain, RocAucTest])

LogReg_MeanScores = [LogReg_AccMean, LogReg_PrecisionMean, LogReg_RecallMean, 
                     LogReg_F1Mean, LogReg_RocAucMean]

LogReg_StdScores = [LogReg_AccStd, LogReg_PrecisionStd, LogReg_RecallStd,
                    LogReg_F1Std, LogReg_RocAucStd]

print("########################################################################")
print("# LogReg Evaluation values #############################################")
print("########################################################################")
print("ACC mean score = {}".format(LogReg_AccMean))
print("ACC std  = {}".format(LogReg_AccStd))
print("########################################################################")
print("PRECISION mean score = {}".format(LogReg_PrecisionMean))
print("PRECISION std  = {}".format(LogReg_PrecisionStd))
print("########################################################################")
print("RECALL mean score = {}".format(LogReg_RecallMean))
print("RECALL std  = {}".format(LogReg_RecallStd))
print("########################################################################")
print("F1 mean score = {}".format(LogReg_F1Mean))
print("F1 std  = {}".format(LogReg_F1Std))
print("########################################################################")
print("ROC AUC mean score = {}".format(LogReg_RocAucMean))
print("ROC AUC std  = {}".format(LogReg_RocAucStd))





