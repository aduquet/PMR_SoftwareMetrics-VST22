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

df = pd.read_csv('DS_java.csv', index_col=0)

argDT = [
    'byte',
    'short',
    'int',
    'long',
    'double',
    'char',
    'boolean',
    'Integer',
    'float',
    'Byte',
    'Short',
    'Long',
    'Double',
    'Char',
    'Boolean'
    'array_byte', 
    'array_short',
    'array_int',
    'array_long',
    'array_double',
    'array_char',
    'array_boolean',
    'array_Integer',
    'array_float',
    'array_Byte',
    'array_Short',
    'array_Long',
    'array_Double',
    'array_Char',
    'array_Boolean'
]

argDataType = [
    'argDT_byte',
    'argDT_short',
    'argDT_int',
    'argDT_long',
    'argDT_double',
    'argDT_char',
    'argDT_boolean',
    'argDT_Integer',
    'argDT_float',
    'argDT_Byte',
    'argDT_Short',
    'argDT_Long',
    'argDT_Double',
    'argDT_Char',
    'argDT_Boolean'
    'argDT_array_byte', 
    'argDT_array_short',
    'argDT_array_int',
    'argDT_array_long',
    'argDT_array_double',
    'argDT_array_char',
    'argDT_array_boolean',
    'argDT_array_Integer',
    'argDT_array_float',
    'argDT_array_Byte',
    'argDT_array_Short',
    'argDT_array_Long',
    'argDT_array_Double',
    'argDT_array_Char',
    'argDT_array_Boolean',
    'argDT_none'
]

returnDT = [
    'returnDT_byte',
    'returnDT_short',
    'returnDT_int',
    'returnDT_long',
    'returnDT_double',
    'returnDT_char',
    'returnDT_boolean',
    'returnDT_Integer',
    'returnDT_float',
    'returnDT_Byte',
    'returnDT_Short',
    'returnDT_Long',
    'returnDT_Double',
    'returnDT_Char',
    'returnDT_Boolean'
    'returnDT_array_byte', 
    'returnDT_array_short',
    'returnDT_array_int',
    'returnDT_array_long',
    'returnDT_array_double',
    'returnDT_array_char',
    'returnDT_array_boolean',
    'returnDT_array_Integer',
    'returnDT_array_float',
    'returnDT_array_Byte',
    'returnDT_array_Short',
    'returnDT_array_Long',
    'returnDT_array_Double',
    'returnDT_array_Char',
    'returnDT_array_Boolean',
    'returnDT_none'
]

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

MRnames = ['MR_ADD', 'MR_MUL', 'MR_PER', 'MR_INC', 'MR_EXC', 'MR_INV']
mainFeatures = [
    'cyclomatic_complexity',
    'tloc', 
    'sloc_whbl', 
    'sloc_statements_wc',
    'nloc_whbl',  
    'nloc', 
    'token_count', 
    'start_line', 
    'end_line', 
    'numArg', 
    'numLoops', 
    'numVariablesDeclared', 
    'numAritOper', 
    'numExternalMethods',
    'return',
    'Total_return',
    'TotalVariablesReturned',
    'argDT_byte',
    'argDT_short',
    'argDT_int',
    'argDT_long',
    'argDT_double',
    'argDT_char',
    'argDT_boolean',
    'argDT_Integer',
    'argDT_float',
    'argDT_Byte',
    'argDT_Short',
    'argDT_Long',
    'argDT_Double',
    'argDT_Char',
    'argDT_Boolean'
    'argDT_array_byte', 
    'argDT_array_short',
    'argDT_array_int',
    'argDT_array_long',
    'argDT_array_double',
    'argDT_array_char',
    'argDT_array_boolean',
    'argDT_array_Integer',
    'argDT_array_float',
    'argDT_array_Byte',
    'argDT_array_Short',
    'argDT_array_Long',
    'argDT_array_Double',
    'argDT_array_Char',
    'argDT_array_Boolean',
    'argDT_none',
    'returnDT_byte',
    'returnDT_short',
    'returnDT_int',
    'returnDT_long',
    'returnDT_double',
    'returnDT_char',
    'returnDT_boolean',
    'returnDT_Integer',
    'returnDT_float',
    'returnDT_Byte',
    'returnDT_Short',
    'returnDT_Long',
    'returnDT_Double',
    'returnDT_Char',
    'returnDT_Boolean'
    'returnDT_array_byte', 
    'returnDT_array_short',
    'returnDT_array_int',
    'returnDT_array_long',
    'returnDT_array_double',
    'returnDT_array_char',
    'returnDT_array_boolean',
    'returnDT_array_Integer',
    'returnDT_array_float',
    'returnDT_array_Byte',
    'returnDT_array_Short',
    'returnDT_array_Long',
    'returnDT_array_Double',
    'returnDT_array_Char',
    'returnDT_array_Boolean',
    'returnDT_none',
    'ext', 'MR_ADD', 'MR_MUL', 'MR_PER', 'MR_INC', 'MR_EXC', 'MR_INV']

data = df.copy()

for i in returnDT:
    data[i] = 0

for i in argDataType:
    data[i] = 0

for index, row in df.iterrows():
    argDTaux = row['argDT']
    argDTaux = argDTaux.replace(' ','')
    argDTaux = argDTaux.replace('[','')
    argDTaux = argDTaux.replace(']','')
    argDTaux = argDTaux.replace("'",'')

    if argDTaux.find(',') == -1:
        idName = 'argDT_' + argDTaux
        data.at[index,idName] += 1
    else:
        aux = argDTaux.split(',')
        for i in aux:
            idName = 'argDT_'  + i
            data.at[index, idName] += 1

for index, row in df.iterrows():
    argDTaux = str(row['returnType'])
    #print(argDTaux)
    argDTaux = argDTaux.replace(' ','')

    #argDTaux = argDTaux.replace('[','')
    #argDTaux = argDTaux.replace(']','')
    #argDTaux = argDTaux.replace("'",'')

    if argDTaux.find(',') == -1:
        idName = 'returnDT_' + argDTaux
        data.at[index,idName] += 1
    else:
        aux = argDTaux.split(',')
        for i in aux:
            idName = 'returnDT_'  + i
            data.at[index, idName] += 1

data = data[mainFeatures]

mr_add = data.drop(['MR_MUL', 'MR_PER', 'MR_INC', 'MR_EXC', 'MR_INV'], axis=1)
mr_mul = data.drop(['MR_ADD', 'MR_PER', 'MR_INC', 'MR_EXC', 'MR_INV'], axis=1)
mr_per = data.drop(['MR_MUL', 'MR_ADD', 'MR_INC', 'MR_EXC', 'MR_INV'], axis=1)
mr_inc = data.drop(['MR_MUL', 'MR_PER', 'MR_ADD', 'MR_EXC', 'MR_INV'], axis=1)
mr_exc = data.drop(['MR_MUL', 'MR_PER', 'MR_INC', 'MR_ADD', 'MR_INV'], axis=1)
mr_inv = data.drop(['MR_MUL', 'MR_PER', 'MR_INC', 'MR_EXC', 'MR_ADD'], axis=1)



mr_add.to_csv('MR_ADD.csv')
mr_mul.to_csv('MR_MUL.csv')
mr_per.to_csv('MR_PER.csv')
mr_inc.to_csv('MR_INC.csv')
mr_exc.to_csv('MR_EXC.csv')
mr_inv.to_csv('MR_INV.csv')

data.to_csv('prueba.csv')
#print(data.keys())