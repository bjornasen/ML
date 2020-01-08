from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas 

#%matplotlib inline

from sklearn import datasets

#Iris Dataset
#iris = datasets.load_iris()
#X = iris.data
#Y  =iris.target

names  =['KreditFF','Insolvent','Reglerad','man','ålder','pay'] 
dataset = pandas.read_csv('RAWDATA191119.csv', names=names)
array = dataset.values
#X = dataset.iloc[:,0:5]
#y = dataset.iloc[:,5]

X = array[:,0:5]
Y = array[:,5]


print(X[:, 0])


#print(Y)

#KMeans
km = KMeans(n_clusters=3)
km.fit(X)
km.predict(X)
labels = km.labels_
#Plotting
fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(X[:, 0], X[:, 3], X[:, 4],
          c=labels.astype(np.float), edgecolor="k", s=50)
ax.set_xlabel("KreditFF")
ax.set_ylabel("Insolvent")
ax.set_zlabel("Reglerad")
plt.title("K Means", fontsize=14)
plt.show()