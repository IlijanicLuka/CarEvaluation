"""
@author: Luka Ilijanic
"""

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


file_name = "C:/Users/Luka/Desktop/archive/car_evaluation.csv" # https://www.kaggle.com/datasets/elikplim/car-evaluation-data-set
variable_names = ["buying_price", "maint_cost", "no_doors", "no_persons", "lug_boot", "safety", "decision"]
data = pd.read_csv(file_name, header = None, names = variable_names)
print(data)
print(data.dtypes)

def bar_graph(data1, column_name):
    table = pd.crosstab(index = data1[column_name], columns="count")
    table.plot.bar()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    plt.xticks(rotation=0)
    plt.legend()
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.bottom'] = True
    plt.grid(axis = 'y')
    plt.rc('axes', axisbelow=True)
    return table

table_decision = bar_graph(data,"decision")
print(table_decision)
table_buying_price = bar_graph(data,"buying_price")
print(table_buying_price)
table_maint_cost = bar_graph(data,"maint_cost")
print(table_maint_cost)
table_no_doors = bar_graph(data,"no_doors")
print(table_no_doors)
table_no_persons = bar_graph(data,"no_persons")
print(table_no_persons)
table_lug_boot = bar_graph(data,"lug_boot")
print(table_lug_boot)
table_safety = bar_graph(data,"safety")
print(table_safety)

def paired_bar_graph(data1,column_name1,column_name2):
    table = pd.crosstab(index = data1[column_name1], columns = data1[column_name2])
    table.plot.bar()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    plt.xticks(rotation=0)
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.bottom'] = True
    plt.grid(axis = 'y')
    plt.rc('axes', axisbelow=True)
    plt.legend(bbox_to_anchor=(1.04, 0.7), loc="upper left")
    return table

table_bprice_decision = paired_bar_graph(data,"buying_price","decision")
print(table_bprice_decision)   
table_mcost_decision = paired_bar_graph(data,"maint_cost","decision")
print(table_mcost_decision)   
table_nodoors_decision = paired_bar_graph(data,"no_doors","decision")
print(table_nodoors_decision)
table_nopersons_decision = paired_bar_graph(data,"no_persons","decision")
print(table_nopersons_decision)   
table_lboot_decision = paired_bar_graph(data,"lug_boot","decision")
print(table_lboot_decision)
table_safety_decision = paired_bar_graph(data,"safety","decision")
print(table_safety_decision)



test_bprice = stats.chi2_contingency(table_bprice_decision)
print(test_bprice)
test_mcost = stats.chi2_contingency(table_mcost_decision)
print(test_mcost)
test_nodoors = stats.chi2_contingency(table_nodoors_decision)
print(test_nodoors)
test_nopersons = stats.chi2_contingency(table_nopersons_decision)
print(test_nopersons)
test_lug_boot = stats.chi2_contingency(table_lboot_decision)
print(test_lug_boot)
test_safety = stats.chi2_contingency(table_safety_decision)
print(test_safety)


map_decision = {'unacc':0,'acc':1,'good':2,'vgood':3}
map_bprice = {'low':0, 'med':1, 'high':2, 'vhigh':3}
map_mcost = {'low':0, 'med':1, 'high':2, 'vhigh':3}
map_nodoors = {'2':0, '3':1, '4':2, '5more':3}
map_nopersons = {'2':0, '4':1, 'more':2}
map_lugboot = {'small':0, 'med':1, 'big':2}
map_safety = {'low':0, 'med':1, 'high':2}

coded_data = pd.DataFrame()
coded_data['buying_price'] = data['buying_price'].map(map_bprice)
coded_data['maint_cost'] = data['maint_cost'].map(map_mcost)
coded_data['no_doors'] = data['no_doors'].map(map_nodoors)
coded_data['no_persons'] = data['no_persons'].map(map_nopersons)
coded_data['lug_boot'] = data['lug_boot'].map(map_lugboot)
coded_data['safety'] = data['safety'].map(map_safety)
coded_data['decision'] = data['decision'].map(map_decision)

coded_data.to_csv(r'C:/Users/Luka/Desktop/archive/car_evaluation_coded.csv', index = False)

corr = coded_data.corr()
print(corr)
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, linewidth=0.5)
plt.xticks(rotation=30)



