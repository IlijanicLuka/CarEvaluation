"""
@author: Luka Ilijanic
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ROC AUC CV5
log_reg_cv5 = 0.93082
sgd_cv5 = 0.92417
knn_cv5 = 0.99161
radiusN_cv5 = 0.99371
mlp_cv5 = 0.9969
dec_tree_cv5 = 0.9953
extra_tree_cv5 = 0.98966
ensemble_cv5 = 0.99537
# ROC AUC STD CV5
log_reg_std_cv5 = 9.27e-4
sgd_std_cv5 = 3.674e-4
knn_std_cv5 = 22.701e-4
radiusN_std_cv5 = 7.236e-4
mlp_std_cv5 = 17.89e-4
dec_tree_std_cv5 = 26.12e-4
extra_tree_std_cv5 = 84.959e-4
ensemble_std_cv5 = 21.44e-4


# ROC AUC CV10
log_reg_cv10 = 0.93113
sgd_cv10 = 0.92454
knn_cv10 = 0.99159
radiusN_cv10 = 0.99386
mlp_cv10 = 0.99662
dec_tree_cv10 = 0.99609
extra_tree_cv10 = 0.98966
ensemble_cv10 = 0.99538
# ROC AUC STD CV10
log_reg_std_cv10 = 5.105e-4
sgd_std_cv10 = 4.682e-4
knn_std_cv10 = 31.445e-4
radiusN_std_cv10 = 7.604e-4
mlp_std_cv10 = 19.082e-4
dec_tree_std_cv10 = 15.966e-4
extra_tree_std_cv10 = 80.627e-4 
ensemble_std_cv10 = 22.46e-4 

algorithms = ['LogReg', 'SGD', 'KNN', 'RadiusN', 'MLP', 'DecTree', 'ExtraTree', 'Ensemble']

ROC_AUC_cv5 = [log_reg_cv5, sgd_cv5, knn_cv5, radiusN_cv5, mlp_cv5, 
        dec_tree_cv5, extra_tree_cv5, ensemble_cv5]
ROC_AUC_std_cv5 = [log_reg_std_cv5, sgd_std_cv5, knn_std_cv5, radiusN_std_cv5,
          mlp_std_cv5, dec_tree_std_cv5, extra_tree_std_cv5, ensemble_std_cv5]

ROC_AUC_cv10 = [log_reg_cv10, sgd_cv10, knn_cv10, radiusN_cv10, mlp_cv10, 
                dec_tree_cv10, extra_tree_cv10, ensemble_cv10]
ROC_AUC_std_cv10 = [log_reg_std_cv10, sgd_std_cv10, knn_std_cv10, radiusN_std_cv10,
                    mlp_std_cv10, dec_tree_std_cv10, extra_tree_std_cv10, ensemble_std_cv10]



def PlotGraph(metricMeanScore, metricBestMeanScore, ylab):
    plt.figure(figsize=(12,6), layout='tight')
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['font.size'] = 12
    x_axis = np.arange(len(algorithms))*4
    roundScore = [round(elem, 3) for elem in metricMeanScore]
    roundBestScore = [round(elem, 3) for elem in metricBestMeanScore]
    bar1 = plt.bar(x_axis -0.7, roundScore, width = 1.4, yerr=ROC_AUC_std_cv5,
                   ecolor='black', capsize=10)
    bar2 = plt.bar(x_axis +0.7, roundBestScore, width = 1.4, color='r', 
                   yerr=ROC_AUC_std_cv10, ecolor='black', capsize=10)
    plt.xticks(x_axis, algorithms)
    plt.ylabel(ylab, fontsize=15)
    color = ('tab:blue', 'r')
    cmap = dict(zip(color,[bar1, bar2]))
    patches = [Patch(color=v, label=k) for v,k in cmap.items()]
    plt.legend(title='Cross validation', labels=['5-fold', '10-fold'], 
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


PlotGraph(ROC_AUC_cv5, ROC_AUC_cv10, ylab='ROC AUC score')


