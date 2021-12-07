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
plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams['legend.handlelength'] = 0.5
plt.rcParams['axes.xmargin'] = 0
df = pd.read_csv('AUC-comparison_V2.csv', index_col=0)
plt.rcParams["figure.autolayout"] = True

metrics = ['SVM: NF-PF [11]', 'SVM: GK [11]', 'SVM: RWK [11]',	'RF: SM', 'DT: SM',	
    'GMB: SM',	'SVM: SM',	'LG: SM']
metrics_legend = ['Top 3 features',	'Top 6 features', 'Top 9 features', 'Top 12 features',	'Top 15 features', 'Top 18 features',	'21 features (all)']

colors = ['#005ecd', '#aaad87', '#dd9343', '#980f00', '#7b8ce0', '#e8d4d4', '#f3bbcc', '#eeaab2']
#,
#    '#e8d4d4', '#f3bbcc', '#eeaab2', '#e69a97', '#b9ba6b', '#d8dc67', '#f7ff61',
#    '#927f80', '#dbb8d5', '#aac3aa', '#009f00', '#924366', '#60276a', '#0042a3']

# plot grouped bar chart


ax=df.plot(x='MR',
        kind='bar',  width=0.9,  rot=0,
        stacked=False)

plt.legend(metrics,loc='upper center', bbox_to_anchor=(0.5, 1.09), 
    fancybox=True, shadow=True, ncol=8)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005),fontsize=6)

#xtick_loc = [0.0, 10, 20, 30 , 40, 50, 60]
#ax.set_xticks(xtick_loc)

#ax.subplots_adjust(bottom=spacing)
plt.xlabel('Metamorphic relation')
plt.ylabel('AUC-ROC')
plt.grid()
plt.show()