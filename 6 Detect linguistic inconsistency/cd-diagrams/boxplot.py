# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as n'
import seaborn as sns

fields = ['Word Embedding Method', 'module_name', 'AUC_ROC']
df_perf = pd.read_csv('wordembed_mcc_acc2.csv', index_col=False)
plt.rcParams["figure.figsize"] = (8, 4)
plt.rcParams["xtick.labelsize"] = 7
sns_plot = sns.boxplot( y=df_perf["AUC-ROC"], x=df_perf["Word Embedding Method"] )
sns_plot.set_xticklabels(sns_plot.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
fig = sns_plot.get_figure()
fig.savefig("wordembedboxauc.pdf")
