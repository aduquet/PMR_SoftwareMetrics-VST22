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

df = pd.read_csv('auc-metrics-feature-importance.csv', index_col=0)


metrics = ['3',	'6', '9', '12',	'15', '18',	'21']
metrics_legend = ['Top 3 features',	'Top 6 features', 'Top 9 features', 'Top 12 features',	'Top 15 features', 'Top 18 features',	'21 features (all)']

colors = ['#005ecd', '#aaad87', '#dd9343', '#980f00', '#7b8ce0', '#6c4cbd', 'm' ]
#,
#    '#e8d4d4', '#f3bbcc', '#eeaab2', '#e69a97', '#b9ba6b', '#d8dc67', '#f7ff61',
#    '#927f80', '#dbb8d5', '#aac3aa', '#009f00', '#924366', '#60276a', '#0042a3']

ax = plt.gca()

for i in range(0, len(metrics)):
    #df.plot(kind='line',x='MR', y=metrics[i], color=colors[i], ax=ax)
    df.plot(kind='scatter', x='MR', y=metrics[i], color=colors[i], ax=ax)
plt.tight_layout(rect=[0,0,0.75,1])
plt.legend(metrics_legend,loc='upper center', bbox_to_anchor=(1.17,1),
    fancybox=True, shadow=True,)
plt.xlabel('Metamorphic relation')
plt.ylabel('AUC-ROC')
plt.grid()
plt.show()