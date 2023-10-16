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
from sklearn.tree import DecisionTreeClassifier


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

model = DecisionTreeClassifier().fit(X_train, y_train)

AccTrain = model.score(X_train, y_train)
AccTest = model.score(X_test, y_test)
DecTree_AccMean = np.mean([AccTrain, AccTest]) 
DecTree_AccStd = np.std([AccTrain, AccTest])

PrecisionTrain = precision_score(y_train, model.predict(X_train), average='weighted')
PrecisionTest = precision_score(y_test, model.predict(X_test), average='weighted')
DecTree_PrecisionMean = np.mean([PrecisionTrain, PrecisionTest]) 
DecTree_PrecisionStd = np.std([PrecisionTrain, PrecisionTest])

RecallTrain = recall_score(y_train, model.predict(X_train), average='weighted')
RecallTest = recall_score(y_test, model.predict(X_test), average='weighted')
DecTree_RecallMean = np.mean([RecallTrain, RecallTest]) 
DecTree_RecallStd = np.std([RecallTrain, RecallTest])

F1Train = f1_score(y_train, model.predict(X_train), average='weighted')
F1Test = f1_score(y_test, model.predict(X_test), average='weighted')
DecTree_F1Mean = np.mean([F1Train, F1Test]) 
DecTree_F1Std = np.std([F1Train, F1Test])

RocAucTrain = roc_auc_score(y_train, model.predict_proba(X_train), average='weighted', multi_class='ovr')
RocAucTest = roc_auc_score(y_test, model.predict_proba(X_test), average='weighted', multi_class='ovr')
DecTree_RocAucMean = np.mean([RocAucTrain, RocAucTest]) 
DecTree_RocAucStd = np.std([RocAucTrain, RocAucTest])

DecTree_MeanScores = [DecTree_AccMean, DecTree_PrecisionMean, DecTree_RecallMean, 
                      DecTree_F1Mean, DecTree_RocAucMean]

DecTree_StdScores = [DecTree_AccStd, DecTree_PrecisionStd, DecTree_RecallStd,
                     DecTree_F1Std, DecTree_RocAucStd]

print("########################################################################")
print("# DecTree Evaluation values ############################################")
print("########################################################################")
print("ACC mean score = {}".format(DecTree_AccMean))
print("ACC std  = {}".format(DecTree_AccStd))
print("########################################################################")
print("PRECISION mean score = {}".format(DecTree_PrecisionMean))
print("PRECISION std  = {}".format(DecTree_PrecisionStd))
print("########################################################################")
print("RECALL mean score = {}".format(DecTree_RecallMean))
print("RECALL std  = {}".format(DecTree_RecallStd))
print("########################################################################")
print("F1 mean score = {}".format(DecTree_F1Mean))
print("F1 std  = {}".format(DecTree_F1Std))
print("########################################################################")
print("ROC AUC mean score = {}".format(DecTree_RocAucMean))
print("ROC AUC std  = {}".format(DecTree_RocAucStd))



