#importing the libarires
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


#data processing
data=pd.read_csv('Mall_Customers.csv')
x=data.iloc[:,[3,4]].values


#how to know the number of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i , random_state=42,init='k-means++')
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.show()

# Fitting K-Means to the dataset
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
#predict the data
y_pred=kmeans.fit_predict(x)


#plotting the clusters
plt.scatter(x[y_pred==0,0],x[y_pred==0,1],s=100,c='red')
plt.scatter(x[y_pred==1,0],x[y_pred==1,1],s=100,c='blue')
plt.scatter(x[y_pred==2,0],x[y_pred==2,1],s=100,c='black')
plt.scatter(x[y_pred==3,0],x[y_pred==3,1],s=100,c='yellow')
plt.scatter(x[y_pred==4,0],x[y_pred==4,1],s=100,c='orange')
plt.show()

