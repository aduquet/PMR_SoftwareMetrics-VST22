import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import metrics


#data = pd.read_csv('MRs_Dataset\MR_ADD.csv', index_col=0)
data = pd.read_csv('DS-LabelledEncoded-SM\MR_ADD.csv', index_col=0)
MR = 'MR_ADD'

labels = data[MR]

data2 = data.copy()
data2.drop([MR,'ext'], axis=1, inplace=True)
#data2.drop([MR], axis=1, inplace=True)

head = data2.columns.values
data2 = np.asarray(data2)
labels = np.asarray(labels)

X_train, X_test, y_train, y_test = train_test_split(data2, labels, train_size= 0.7, random_state=42)


clf=RandomForestClassifier(n_estimators=1000)
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=None, max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0,
#            warm_start=False)
clf.fit(X_train,y_train)

importance = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(importance)[::-1]
indices = np.sort(indices)
print(head)
print(importance, indices)

y_pred=clf.predict(X_test)
df = DataFrame(['featureName', 'importance'])
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

