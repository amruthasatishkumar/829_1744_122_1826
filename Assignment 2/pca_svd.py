import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#%matplotlib inline
url = "C:\Users\pbhav\Desktop\5-offical\DA\iris.csv"
df = pd.read_csv(url)
df.head()

from sklearn.preprocessing import StandardScaler
features=['sepal length','sepal width','petal length','petal width']
target = ['species']
x = df.loc[:,features].values
y = df.loc[:,target].values
x=StandardScaler().fit_transform(x)
pd.DataFrame(data = x, columns = features).head()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents=pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal comp 1','principal comp 2'])
principalDf.head()

target=df['species']
target.head()


import matplotlib.pyplot as plt
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('principal comp 1', fontsize = 15)
ax.set_ylabel('principal comp 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['setosa', 'versicolor', 'virginica']
colors = ['r','g','b']
for target, color in zip(targets,colors):
	indicesToKeep = finalDf['species'] == target
	ax.scatter(finalDf.loc[indicesToKeep, 'principal comp 1'] , finalDf.loc[indicesToKeep, 'principal comp 2'] , c = color, s = 50)
ax.legend(targets)
ax.grid()