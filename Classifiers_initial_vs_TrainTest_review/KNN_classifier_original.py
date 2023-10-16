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

model = KNeighborsClassifier().fit(X_train, y_train)

AccTrain = model.score(X_train, y_train)
AccTest = model.score(X_test, y_test)
KNN_AccMean = np.mean([AccTrain, AccTest]) 
KNN_AccStd = np.std([AccTrain, AccTest])

PrecisionTrain = precision_score(y_train, model.predict(X_train), average='weighted')
PrecisionTest = precision_score(y_test, model.predict(X_test), average='weighted')
KNN_PrecisionMean = np.mean([PrecisionTrain, PrecisionTest]) 
KNN_PrecisionStd = np.std([PrecisionTrain, PrecisionTest])

RecallTrain = recall_score(y_train, model.predict(X_train), average='weighted')
RecallTest = recall_score(y_test, model.predict(X_test), average='weighted')
KNN_RecallMean = np.mean([RecallTrain, RecallTest]) 
KNN_RecallStd = np.std([RecallTrain, RecallTest])

F1Train = f1_score(y_train, model.predict(X_train), average='weighted')
F1Test = f1_score(y_test, model.predict(X_test), average='weighted')
KNN_F1Mean = np.mean([F1Train, F1Test]) 
KNN_F1Std = np.std([F1Train, F1Test])

RocAucTrain = roc_auc_score(y_train, model.predict_proba(X_train), average='weighted', multi_class='ovr')
RocAucTest = roc_auc_score(y_test, model.predict_proba(X_test), average='weighted', multi_class='ovr')
KNN_RocAucMean = np.mean([RocAucTrain, RocAucTest]) 
KNN_RocAucStd = np.std([RocAucTrain, RocAucTest])

KNN_MeanScores = [KNN_AccMean, KNN_PrecisionMean, KNN_RecallMean, 
                  KNN_F1Mean, KNN_RocAucMean]

KNN_StdScores = [KNN_AccStd, KNN_PrecisionStd, KNN_RecallStd,
                 KNN_F1Std, KNN_RocAucStd]

print("########################################################################")
print("# KNN Evaluation values ################################################")
print("########################################################################")
print("ACC mean score = {}".format(KNN_AccMean))
print("ACC std  = {}".format(KNN_AccStd))
print("########################################################################")
print("PRECISION mean score = {}".format(KNN_PrecisionMean))
print("PRECISION std  = {}".format(KNN_PrecisionStd))
print("########################################################################")
print("RECALL mean score = {}".format(KNN_RecallMean))
print("RECALL std  = {}".format(KNN_RecallStd))
print("########################################################################")
print("F1 mean score = {}".format(KNN_F1Mean))
print("F1 std  = {}".format(KNN_F1Std))
print("########################################################################")
print("ROC AUC mean score = {}".format(KNN_RocAucMean))
print("ROC AUC std  = {}".format(KNN_RocAucStd))



