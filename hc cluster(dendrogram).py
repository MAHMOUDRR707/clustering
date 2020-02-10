#importing the libarires
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


#data processing
data=pd.read_csv('Mall_Customers.csv')
x=data.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(x,method='ward'))

#how to know the number of clusters
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
#predict the data
y_pred=hc.fit_predict(x)


#plotting the clusters
plt.scatter(x[y_pred==0,0],x[y_pred==0,1],s=100,c='red')
plt.scatter(x[y_pred==1,0],x[y_pred==1,1],s=100,c='blue')
plt.scatter(x[y_pred==2,0],x[y_pred==2,1],s=100,c='black')
plt.scatter(x[y_pred==3,0],x[y_pred==3,1],s=100,c='yellow')
plt.scatter(x[y_pred==4,0],x[y_pred==4,1],s=100,c='orange')
plt.show()

