import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('finalFeatureImportance.csv')

print(len(df.keys()), df.keys())

data = df.drop(df.columns[df.apply(lambda col: col.sum() == 0)], axis=1)
data.to_csv('final2_featureImportance.csv')
#data.drop(columns=['MR'], axis=1)

metrics = ['cyclomatic_complexity', 'tloc', 'sloc_whbl', 
    'sloc_statements_wc', 'nloc_whbl', 'nloc', 'token_count', 'start_line',
    'end_line', 'numArg', 'numLoops', 'numVariablesDeclared', 'numAritOper',
    'numExternalMethods', 'return', 'Total_return',
    'TotalVariablesReturned', 'argDT_byte', 'argDT_short', 'argDT_int',
    'argDT_long', 'argDT_double', 'argDT_char', 'argDT_boolean',
    'argDT_Integer', 'argDT_float', 'argDT_Byte', 'argDT_Short',
    'argDT_Long', 'argDT_Double', 'argDT_Char']

colors = ['#565656', '#585858', '#5a5a5a', '#5c5c5c', '#5e5e5e', '#60605f', '#616261', '#636563', '#656764', '#676966', '#686b67', '#6a6e68', '#6b7069', '#6d726a', '#6e756b', '#6f776c', '#707a6d', '#717c6d', '#727f6d', '#73826d', '#73856d', '#74876c', '#748a6b', '#738d6a', '#739068', '#719465', '#6f9762', '#6d9b5e', '#689e58', '#61a34f', '#54a841', '#13b000']
print(len(data.keys()), data.keys())

#for index, row in df.iterrows():
    #print(index)
#    print(row)
#df.drop(df.columns == 0, axis=1)

#print(df.keys(), len(df.keys()))

ax = plt.gca()

for i in range(0, len(metrics)):

    df.plot(kind='line',x='MR', y=metrics[i], color=colors[i], ax=ax)

plt.grid()
plt.show()