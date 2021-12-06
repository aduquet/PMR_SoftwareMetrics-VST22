import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fsize = 10
tsize = 10

tdir = 'in'

major = 5.0
minor = 3.0

style = 'default'
plt.rcParams["font.family"] = "Times New Roman"
plt.style.use(style)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = fsize
plt.rcParams['legend.fontsize'] = 8
#plt.rcParams['xtick.direction'] = tdir
#plt.rcParams['ytick.direction'] = tdir
#plt.rcParams['xtick.major.size'] = major
#plt.rcParams['xtick.minor.size'] = minor
#plt.rcParams['ytick.major.size'] = major
#plt.rcParams['ytick.minor.size'] = minor
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams['legend.handlelength'] = 0.5
df = pd.read_csv('final3_featureImportance-ABC.csv', index_col=0)

#print(len(df.keys()), df.keys())

#data = df.drop(df.columns[df.apply(lambda col: col.sum() == 0)], axis=1)
#data.to_csv('final2_featureImportance.csv')
#data.drop(columns=['MR'], axis=1)

#metrics = ['cyclomatic_complexity', 'tloc', 'sloc_whbl', 
#    'sloc_statements_wc', 'nloc_whbl', 'nloc', 'token_count', 'start_line',
#    'end_line', 'numArg', 'numLoops', 'numVariablesDeclared', 'numAritOper',
#    'numExternalMethods', 'return', 'Total_return',
#    'TotalVariablesReturned', 'argDT_byte', 'argDT_short', 'argDT_int',
#    'argDT_long', 'argDT_double', 'argDT_char', 'argDT_boolean',
#    'argDT_Integer', 'argDT_float', 'argDT_Byte', 'argDT_Short',
#    'argDT_Long', 'argDT_Double', 'argDT_Char']

#metrics = ['CCN', 'tloc', 'sloc-whbl', 'sloc-statements', 'nloc-whbl',
#    'nloc', 'token-count', 'start-line', 'end-line', 'numArg', 'numLoops',
#    'totalVar', 'numOper', 'numMethCall', 'hasReturn', 'totalReturn',
#    'numOperands', 'dataArg', 'returnDataType', 'ext', 'full-Parameters']

metrics = ['A: dataArg', 'B: CCN', 'C: tloc', 'D: sloc-whbl', 'E: sloc-statements', 'F: nloc-whbl',
    'G: nloc', 'H: token-count', 'I: start-line', 'J: end-line', 'K: numArg', 'L: numLoops',
    'M: totalVar', 'N: numOper', 'O: numMethCall', 'P: hasReturn', 'Q: totalReturn',
    'R: numOperands', 'S: returnDataType', 'T: ext', 'U: full-Parameters']


colors = ['#005ecd', '#aaad87', '#dd9343', '#980f00', '#7b8ce0', '#6c4cbd', '#93003a',
    '#e8d4d4', '#f3bbcc', '#eeaab2', '#e69a97', '#b9ba6b', '#d8dc67', '#f7ff61',
    '#927f80', '#dbb8d5', '#aac3aa', '#009f00', '#924366', '#60276a', '#0042a3']

#for index, row in df.iterrows():
    #print(index)
#    print(row)
#df.drop(df.columns == 0, axis=1)

#print(df.keys(), len(df.keys()))

ax = plt.gca()

for i in range(0, len(metrics)):
    df.plot(kind='line',x='MR', y=metrics[i], color=colors[i], ax=ax)
    df.plot(kind='scatter', x='MR', y=metrics[i], color=colors[i], ax=ax)
plt.tight_layout(rect=[0,0,0.75,1])
plt.legend(metrics,loc='upper center', bbox_to_anchor=(1.17,1),
    fancybox=True, shadow=True,)
plt.xlabel('Metamorphic relation')
plt.ylabel('Feature importance')
plt.grid()
plt.show()