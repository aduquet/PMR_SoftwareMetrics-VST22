import pickle
import warnings
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

warnings.filterwarnings('ignore')

plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 10})


if __name__ == '__main__':  
    import click

    @click.command()
    @click.option('-i', '--file', 'data', help='Path of labelled Dataset')
    @click.option('-ns', '--nSplits', 'ns', help='Number of splits')

    def main(data, ns):
  
        df = pd.read_csv(data, index_col=0)
        colNames = list(df.columns.values)
        #print(df)
         
        for i in colNames:
            if i.find('MR_') != -1:
                MRname = i

        labels = df[MRname]
        data = df.copy()
        data.drop(MRname, axis=1, inplace=True)

        data.fillna(0)

        #data.get_dummies(obj_df, columns=["body_style", "drive_wheels"], prefix=["body", "drive"])
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,random_state=109) # 70% training and 30% test



        random_state = np.random.RandomState(0)

        SVM = SVC(C=1000, probability=True, random_state=random_state)

        clf = svm.SVC(kernel='linear') # Linear Kernel
        rf = RandomForestClassifier(n_estimators=100)

        #Train the model using the training sets
        clf.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        #Predict the response for test dataset
        y_pred = clf.predict(X_test)
        y_pred_rf= rf.predict(X_test)

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
        print(fpr, tpr)
        print(roc_auc_score(y_test, y_pred, average=None))

        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# python training.py -i "C:\Users\duquet\Documents\GitHub\RENE-PredictingMetamorphicRelations\Phase_II-DataPreparation\Labelled-Dataset\RWK_DS_JK.csv" -ns 10 
        print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
        print("Recall:",metrics.recall_score(y_test, y_pred))
        print('***')

        print("Accuracy:",metrics.accuracy_score(y_test, y_pred_rf))

        print('----')
        print(y_pred)
        print(y_test)
main()
