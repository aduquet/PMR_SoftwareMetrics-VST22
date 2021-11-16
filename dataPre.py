import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.svm import SVC
from sklearn.metrics import *
from scipy import interp
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 10})

class CSminerFeatures():
    metric_names = [ 
        'cyclomatic_complexity',
        'tloc', 
        'sloc_whbl', 
        'sloc_statements_wc',
        'nloc_whbl',  
        'nloc', 
        'token_count', 
        'start_line', 
        'end_line', 
        'full_parameters',
        'argDT', 
        'numArg', 
        'numLoops', 
        'numVariablesDeclared', 
        'numAritOper', 
        'numExternalMethods',
        'return',
        'Total_return',
        'TotalVariablesReturned',
        'returnType',
        'ext'
    ]   


def dfPrep(df, i):
    metric_names = [ 
        'cyclomatic_complexity',
        'tloc', 
        'sloc_whbl', 
        'sloc_statements_wc',
        'nloc_whbl',  
        'nloc', 
        'token_count', 
        'start_line', 
        'end_line', 
        'full_parameters',
        'argDT', 
        'numArg', 
        'numLoops', 
        'numVariablesDeclared', 
        'numAritOper', 
        'numExternalMethods',
        'return',
        'Total_return',
        'TotalVariablesReturned',
        'returnType',
        'ext']    
    metric_names.append(i)
    df2 = df[metric_names]
    return df2

if __name__ == '__main__':  
    import click

    @click.command()
    @click.option('-i', '--file', 'data', help='Path of labelled Dataset')
    @click.option('-ns', '--nSplits', 'ns', help='Number of splits')

    def main(data, ns):
        
        mainPath_name = data.split('\\')[-1]
        mainPath_name = mainPath_name.split('.')[0]
  
        df = pd.read_csv(data, index_col=0)
        colNames = list(df.columns.values)
        MRnames = []
        
        for i in colNames:
            if i.find('MR_') != -1:
                MRnames.append(i)

        for i in MRnames:
            df2 = dfPrep(df, i)
            #df2 = df2.to_csv(i + '.csv')

            data=np.asarray(df2)
            print(data)

        random_state = np.random.RandomState(0)

        SVM = SVC(C=1000, probability=True, random_state=random_state)
        skf = StratifiedKFold(n_splits=int(ns))


# python training.py -i "C:\Users\duquet\Documents\GitHub\RENE-PredictingMetamorphicRelations\Phase_II-DataPreparation\Labelled-Dataset\RWK_DS_JK.csv" -ns 10 

main()
