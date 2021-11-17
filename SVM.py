import pickle
import warnings
from networkx.drawing.nx_pylab import draw_networkx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import *
from scipy import interp
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

#mr = pd.read_csv('MR_ADD.csv', index_col=0)
#MR = 'MR_ADD'

#mr = pd.read_csv('MR_MUL.csv', index_col=0)
#MR = 'MR_MUL'

#mr = pd.read_csv('MR_PER.csv', index_col=0)
#MR = 'MR_PER'

#mr = pd.read_csv('MR_INC.csv', index_col=0)
#MR = 'MR_INC'

#mr = pd.read_csv('MR_EXC.csv', index_col=0)
#MR = 'MR_EXC'

#mr = pd.read_csv('MR_INV.csv', index_col=0)
#MR = 'MR_INV'

model = 'SVM'
output = model + '_' + MR

labels = mr[MR]
data = mr.copy()
data.drop([MR,'ext'], axis=1, inplace=True)

data = np.asarray(data)
labels = np.asarray(labels)

skf = StratifiedKFold(n_splits=10)

metric = [
    'fold',
    'TN',
    'FN', 
    'TP',
    'FP',
    'TPR',
    'FPR',
    'TNR',
    'FNR',
    'recall',
    'acc',
    'f1',
    'precision_w',
    'precision_macro',
    'precision_micro',
    'bsr',
    'aucM'
]

score = []
count = 0
for train_index, test_index in skf.split(data, labels):

    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    #X_train, X_test, y_train, y_test = train_test_split(add, labels, test_size=0.3, random_state=47) # 70% training and 30% test

    rf = svm.SVC(kernel='linear') 
    rf.fit(x_train, y_train)
    y_pred_rf= rf.predict(x_test)

    # METRICS
    CM = confusion_matrix(y_test, y_pred_rf)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    TNR = 1 - FPR
    FNR = FN/(FN+TP)

    recall = metrics.recall_score(y_test, y_pred_rf)
    acc = metrics.accuracy_score(y_test, y_pred_rf)
    f1 = metrics.f1_score(y_test, y_pred_rf)
    precision_w = precision_score(y_test, y_pred_rf, average='weighted')
    precision_macro = precision_score(y_test, y_pred_rf, average='macro')
    precision_micro = precision_score(y_test, y_pred_rf, average='micro')
    bsr = balanced_accuracy_score(y_test, y_pred_rf)
    aucM = roc_auc_score(y_test, y_pred_rf)
    count += 1
    fold = 'k_'+str(count) 
    score.append([fold, TN, FN, TP, FP, TPR, FPR, TNR, FNR, recall, acc, f1, precision_w, precision_macro, precision_micro, bsr, aucM])
   
    #print("Recall:",metrics.recall_score(y_test, y_pred_rf))
    #print('***')
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred_rf))
    #print('***')
    #print("F1:", metrics.f1_score(y_test, y_pred_rf))
    #print('***')
    #print("Presicion:", precision_score(y_test, y_pred_rf, average='weighted'))
    #print('***')
    #print("BSR:", balanced_accuracy_score(y_test, y_pred_rf))
    #print('***')
    #print("AUC:", roc_auc_score(y_test, y_pred_rf))      

df_score = pd.DataFrame(score, columns = metric)

df_score.to_csv(output + '.csv')