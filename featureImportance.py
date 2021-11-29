import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import *


#data = pd.read_csv('MRs_Dataset\MR_ADD.csv', index_col=0)
data = pd.read_csv('DS-LabelledEncoded-SM\MR_ADD.csv', index_col=0)
data = pd.read_csv('DS-LabelledEncoded-SM\MR_ADD.csv', index_col=0)
data = pd.read_csv('DS-LabelledEncoded-SM\MR_ADD.csv', index_col=0)
data = pd.read_csv('DS-LabelledEncoded-SM\MR_ADD.csv', index_col=0)
data = pd.read_csv('DS-LabelledEncoded-SM\MR_ADD.csv', index_col=0)
MR = 'MR_ADD'

labels = data[MR]

data2 = data.copy()
data2.drop([MR,'ext'], axis=1, inplace=True)
#data2.drop([MR], axis=1, inplace=True)

head = data2.columns.values
#print(len(head))
#data2 = np.asarray(data2)
#labels = np.asarray(labels)

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

featureImportanceScore_l = ['cyclomatic_complexity', 'tloc', 'sloc_whbl', 'sloc_statements_wc',
    'nloc_whbl', 'nloc', 'token_count', 'start_line', 'end_line', 'numArg', 'numLoops', 'numVariablesDeclared', 
    'numAritOper', 'numExternalMethods', 'return', 'Total_return', 'TotalVariablesReturned', 'argDT_byte', 'argDT_short', 
    'argDT_int', 'argDT_long', 'argDT_double', 'argDT_char', 'argDT_boolean', 'argDT_Integer', 'argDT_float', 'argDT_Byte', 'argDT_Short', 'argDT_Long', 'argDT_Double', 'argDT_Char', 'argDT_BooleanargDT_array_byte', 'argDT_array_short', 'argDT_array_int', 'argDT_array_long',
    'argDT_array_double', 'argDT_array_char', 'argDT_array_boolean', 'argDT_array_Integer', 'argDT_array_float', 'argDT_array_Byte', 'argDT_array_Short', 'argDT_array_Long', 'argDT_array_Double', 'argDT_array_Char', 'argDT_array_Boolean', 'argDT_none' 'returnDT_byte', 
    'returnDT_short', 'returnDT_int', 'returnDT_long', 'returnDT_double', 'returnDT_char', 'returnDT_boolean', 'returnDT_Integer', 'returnDT_float', 'returnDT_Byte', 'returnDT_Short', 'returnDT_Long', 'returnDT_Double', 'returnDT_Char', 'returnDT_BooleanreturnDT_array_byte', 'returnDT_array_short', 'returnDT_array_int',
    'returnDT_array_long', 'returnDT_array_double', 'returnDT_array_char', 'returnDT_array_boolean', 'returnDT_array_Integer', 'returnDT_array_float', 'returnDT_array_Byte', 'returnDT_array_Short', 'returnDT_array_Long', 'returnDT_array_Double', 'returnDT_array_Char', 'returnDT_array_Boolean', 'returnDT_none'] 

clf= SelectFromModel(RandomForestClassifier(n_estimators=1000))
for i in range(0,1):
    X_train, X_test, y_train, y_test = train_test_split(data2, data[MR], train_size= 0.7, random_state=42)
    clf.fit(X_train,y_train)
    #importance = clf.feature_importances_
    #std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    #indices = np.argsort(importance)[::-1]
    #indices = np.sort(indices)
    clf.get_support()
    selected_feat= X_train.columns[(clf.get_support())]
    #len(selected_feat)
    #print(selected_feat)
    #print(clf.feature_importances_)

score = []
featureScore = []
indeceScore = []
clf= RandomForestClassifier(n_estimators=1000)

def sort(importance, indices):

    featureimportanceScore = []
    #print(importance)
    for i in range(0,len(indices)):
        a = indices[i]
        #print(a)
        #print(importance[a])
        featureimportanceScore.append(importance[a])
   # print(len(featureimportanceScore))
    return featureimportanceScore

featureimportanceScore_aux = []
for i in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(data2, data[MR], train_size= 0.7, random_state=42)
    clf.fit(X_train,y_train)
    y_pred_rf= clf.predict(X_test)

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
    fold = 'k_'+ str(i) 
    score.append([fold, TN, FN, TP, FP, TPR, FPR, TNR, FNR, recall, acc, f1, precision_w, precision_macro, precision_micro, bsr, aucM])
    
    importance = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

    indices = np.argsort(importance)[::-1]
    #indices = np.sort(indices)

    featureImportanceList = sort(importance, indices)
    featureimportanceScore_aux.append(featureImportanceList)
    print(len(featureimportanceScore_aux))
    featureScore.append(importance)
    indeceScore.append(indices)
    #print(featureImportanceList)
#print(len(featureImportanceScore_l))
#print(featureimportanceScore_aux, len(featureimportanceScore_aux))
#print(score, len(score))

df_score = pd.DataFrame(score, columns = metric)
df_featureimportance = pd.DataFrame(featureimportanceScore_aux, columns = head)
df_score.to_csv('RF_metrics.csv')
df_featureimportance.to_csv('featureImportance.csv')
    #importance = clf.feature_importances_
    #std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    #indices = np.argsort(importance)[::-1]
    #indices = np.sort(indices)

#pd.series(clf.estimator_,clf.feature_importances_,clf.ravel()).hist()
#print(head)
#print(importance, indices)

#y_pred=clf.predict(X_test)
#df = DataFrame(['featureName', 'importance'])
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


