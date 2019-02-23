import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/suizhehao/mxnet-cv-project/dog_data/labels.csv')
count = data['breed'].value_counts()
s = pd.Series(count)
s.plot(kind='bar',stacked=True)
plt.show()
# print(s)
# print(count)
# print(data.describe())