"""
@author: Luka Ilijanic
"""


import matplotlib.pyplot as plt
from LogReg_classifier_original import LogReg_MeanScores, LogReg_StdScores
from SGD_classifier_original import SGD_MeanScores, SGD_StdScores
from KNN_classifier_original import KNN_MeanScores, KNN_StdScores
from RadiusN_classifier_original import RadiusN_MeanScores, RadiusN_StdScores
from MLP_classifier_original import MLP_MeanScores, MLP_StdScores
from DecTree_classifier_original import DecTree_MeanScores, DecTree_StdScores
from ExtraTree_classifier_original import ExtraTree_MeanScores, ExtraTree_StdScores


algorithms = ['LogReg', 'SGD', 'KNN', 'RadiusN', 'MLP', 'DecTree', 'ExtraTree']


AccMeanScores = [LogReg_MeanScores[0], SGD_MeanScores[0], KNN_MeanScores[0], 
                 RadiusN_MeanScores[0], MLP_MeanScores[0], DecTree_MeanScores[0],
                 ExtraTree_MeanScores[0]]
PrecisionMeanScores = [LogReg_MeanScores[1], SGD_MeanScores[1], KNN_MeanScores[1],
                       RadiusN_MeanScores[1], MLP_MeanScores[1], DecTree_MeanScores[1],
                       ExtraTree_MeanScores[1]]
RecallMeanScores = [LogReg_MeanScores[2], SGD_MeanScores[2], KNN_MeanScores[2],
                    RadiusN_MeanScores[2], MLP_MeanScores[2], DecTree_MeanScores[2],
                    ExtraTree_MeanScores[2]]
F1MeanScores = [LogReg_MeanScores[3], SGD_MeanScores[3], KNN_MeanScores[3],
                RadiusN_MeanScores[3], MLP_MeanScores[3], DecTree_MeanScores[3],
                ExtraTree_MeanScores[3]]
RocAucMeanScores = [LogReg_MeanScores[4], SGD_MeanScores[4], KNN_MeanScores[4],
                    RadiusN_MeanScores[4], MLP_MeanScores[4], DecTree_MeanScores[4],
                    ExtraTree_MeanScores[4]]


AccStdScores = [LogReg_StdScores[0], SGD_StdScores[0], KNN_StdScores[0], 
                RadiusN_StdScores[0], MLP_StdScores[0], DecTree_StdScores[0],
                ExtraTree_StdScores[0]]

PrecisionStdScores = [LogReg_StdScores[1], SGD_StdScores[1], KNN_StdScores[1],
                      RadiusN_StdScores[1], MLP_StdScores[1], DecTree_StdScores[1],
                      ExtraTree_StdScores[1]]

RecallStdScores = [LogReg_StdScores[2], SGD_StdScores[2], KNN_StdScores[2],
                   RadiusN_StdScores[2], MLP_StdScores[2], DecTree_StdScores[2],
                   ExtraTree_StdScores[2]]

F1StdScores = [LogReg_StdScores[3], SGD_StdScores[3], KNN_StdScores[3],
               RadiusN_StdScores[3], MLP_StdScores[3], DecTree_StdScores[3],
               ExtraTree_StdScores[3]]

RocAucStdScores = [LogReg_StdScores[4], SGD_StdScores[4], KNN_StdScores[4],
                   RadiusN_StdScores[4], MLP_StdScores[4], DecTree_StdScores[4],
                   ExtraTree_StdScores[4]]


def PlotGraph(metricMeanScore, ylab):
    plt.figure(figsize=(8,5))
    roundScore = [round(elem, 3) for elem in metricMeanScore]
    bar1 = plt.bar(algorithms, roundScore, align='center')
    plt.ylabel(ylab, fontsize=14)
    plt.bar_label(bar1, padding=3, color='tab:blue')
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.bottom'] = True
    plt.grid(axis = 'y')
    plt.rc('axes', axisbelow=True)
    plt.show()


PlotGraph(AccMeanScores, ylab='ACC score')
PlotGraph(PrecisionMeanScores, ylab='PRECISION score')
PlotGraph(RecallMeanScores, ylab='RECALL score')
PlotGraph(F1MeanScores, ylab='F1 score')
PlotGraph(RocAucMeanScores, ylab='ROC AUC score')



