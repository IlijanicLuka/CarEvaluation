"""
@author: Luka Ilijanic
"""

import numpy as np
import matplotlib.pyplot as plt
from LogReg_TrainTest import LogReg_BestMeanScores, LogReg_BestStdScores, LogReg_BestParams
from SGD_TrainTest import SGD_BestMeanScores, SGD_BestStdScores, SGD_BestParams
from KNN_TrainTest import KNN_BestMeanScores, KNN_BestStdScores, KNN_BestParams
from RadiusN_TrainTest import RadiusN_BestMeanScores, RadiusN_BestStdScores, RadiusN_BestParams
from MLP_TrainTest import MLP_BestMeanScores, MLP_BestStdScores, MLP_BestParams
from DecTree_TrainTest import DecTree_BestMeanScores, DecTree_BestStdScores, DecTree_BestParams
from ExtraTree_TrainTest import ExtraTree_BestMeanScores, ExtraTree_BestStdScores, ExtraTree_BestParams
from Initial_graphs import AccMeanScores, PrecisionMeanScores, RecallMeanScores, F1MeanScores, RocAucMeanScores
from Initial_graphs import AccStdScores, PrecisionStdScores, RecallStdScores, F1StdScores, RocAucStdScores
from matplotlib.patches import Patch


algorithms = ['LogReg', 'SGD', 'KNN', 'RadiusN', 'MLP', 'DecTree', 'ExtraTree']


Acc_BestMeanScores = [LogReg_BestMeanScores[0], SGD_BestMeanScores[0], KNN_BestMeanScores[0], 
                      RadiusN_BestMeanScores[0], MLP_BestMeanScores[0], DecTree_BestMeanScores[0],
                      ExtraTree_BestMeanScores[0]]
Precision_BestMeanScores = [LogReg_BestMeanScores[1], SGD_BestMeanScores[1], KNN_BestMeanScores[1],
                            RadiusN_BestMeanScores[1], MLP_BestMeanScores[1], DecTree_BestMeanScores[1],
                            ExtraTree_BestMeanScores[1]]
Recall_BestMeanScores = [LogReg_BestMeanScores[2], SGD_BestMeanScores[2], KNN_BestMeanScores[2],
                         RadiusN_BestMeanScores[2], MLP_BestMeanScores[2], DecTree_BestMeanScores[2],
                         ExtraTree_BestMeanScores[2]]
F1_BestMeanScores = [LogReg_BestMeanScores[3], SGD_BestMeanScores[3], KNN_BestMeanScores[3],
                     RadiusN_BestMeanScores[3], MLP_BestMeanScores[3], DecTree_BestMeanScores[3],
                     ExtraTree_BestMeanScores[3]]
RocAuc_BestMeanScores = [LogReg_BestMeanScores[4], SGD_BestMeanScores[4], KNN_BestMeanScores[4],
                         RadiusN_BestMeanScores[4], MLP_BestMeanScores[4], DecTree_BestMeanScores[4],
                         ExtraTree_BestMeanScores[4]]


Acc_BestStdScores = [LogReg_BestStdScores[0], SGD_BestStdScores[0], KNN_BestStdScores[0], 
                     RadiusN_BestStdScores[0], MLP_BestStdScores[0], DecTree_BestStdScores[0],
                     ExtraTree_BestStdScores[0]]
Precision_BestStdScores = [LogReg_BestStdScores[1], SGD_BestStdScores[1], KNN_BestStdScores[1],
                           RadiusN_BestStdScores[1], MLP_BestStdScores[1], DecTree_BestStdScores[1],
                           ExtraTree_BestStdScores[1]]
Recall_BestStdScores = [LogReg_BestStdScores[2], SGD_BestStdScores[2], KNN_BestStdScores[2],
                        RadiusN_BestStdScores[2], MLP_BestStdScores[2], DecTree_BestStdScores[2],
                        ExtraTree_BestStdScores[2]]
F1_BestStdScores = [LogReg_BestStdScores[3], SGD_BestStdScores[3], KNN_BestStdScores[3],
                    RadiusN_BestStdScores[3], MLP_BestStdScores[3], DecTree_BestStdScores[3],
                    ExtraTree_BestStdScores[3]]
RocAuc_BestStdScores = [LogReg_BestStdScores[4], SGD_BestStdScores[4], KNN_BestStdScores[4],
                        RadiusN_BestStdScores[4], MLP_BestStdScores[4], DecTree_BestStdScores[4],
                        ExtraTree_BestStdScores[4]]



def PlotGraph(metricMeanScore, metricBestMeanScore, metricStdScore, metricBestStdScore, ylab):
    plt.figure(figsize=(12,6), layout='tight')
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['font.size'] = 12
    x_axis = np.arange(len(algorithms))*4
    roundScore = [round(elem, 3) for elem in metricMeanScore]
    roundBestScore = [round(elem, 3) for elem in metricBestMeanScore]
    bar1 = plt.bar(x_axis -0.7, roundScore, width = 1.4, yerr=metricStdScore,
                   ecolor='black', capsize=10)
    bar2 = plt.bar(x_axis +0.7, roundBestScore, width = 1.4, color='r', 
                   yerr=metricBestStdScore, ecolor='black', capsize=10)
    plt.xticks(x_axis, algorithms)
    plt.ylabel(ylab, fontsize=15)
    color = ('tab:blue', 'r')
    cmap = dict(zip(color,[bar1, bar2]))
    patches = [Patch(color=v, label=k) for v,k in cmap.items()]
    plt.legend(title='Algorithms', labels=['Default hyperparameters', 'Best hyperparameters'], 
               handles=patches, bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)
    plt.bar_label(bar1, padding=3, color='tab:blue')
    plt.bar_label(bar2, padding=3, color='r')
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.bottom'] = True
    plt.grid(axis = 'y')
    plt.rc('axes', axisbelow=True)
    plt.show()



PlotGraph(AccMeanScores, Acc_BestMeanScores, AccStdScores, Acc_BestStdScores, ylab='ACC score')
PlotGraph(PrecisionMeanScores, Precision_BestMeanScores, PrecisionStdScores, Precision_BestStdScores, ylab='PRECISION score')
PlotGraph(RecallMeanScores, Recall_BestMeanScores, RecallStdScores, Recall_BestStdScores, ylab='RECALL score')
PlotGraph(F1MeanScores, F1_BestMeanScores, F1StdScores, F1_BestStdScores, ylab='F1 score')
PlotGraph(RocAucMeanScores, RocAuc_BestMeanScores, RocAucStdScores, RocAuc_BestStdScores, ylab='ROC AUC score')



