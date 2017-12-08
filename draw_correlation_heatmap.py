import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run(TRAINING_DATA, data_name):
  df = pd.read_csv(TRAINING_DATA, header=None)
  corrMatrix = df.corr()
  b = sns.heatmap(corrMatrix, mask=np.zeros_like(corrMatrix, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
              square=True, xticklabels=False, yticklabels=False)
  plt.savefig(
      'correlation_heatmaps/Correlation_Heatmap_{}.png'.format(data_name), bbox_inches='tight')
  plt.clf()
  print('Correlation heatmap saved to correlation_heatmaps/Correlation_Heatmap_{}.png'.format(data_name))
  return
