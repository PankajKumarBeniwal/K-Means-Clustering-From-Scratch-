#Importing all the necessary libraries 
import random as rd
import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
#Reading the data file here the csv file
dataset = pd.read_csv('/home/pankaj/Desktop/OldFaithfulData.csv')
#Preprocessing the data
df = preprocessing.scale(dataset)
data = pd.DataFrame(data = df, columns = ["eruptions", "waiting"])
#Annaul Income and Spending Score only 
X = data.iloc[:, [0, 1]].values

m = X.shape[0] #number of training examples
n = X.shape[1] #number of features. Here n=2
n_iter = 1  #number of iteratins
K = 3  #number of clusters(k = 3)

Centroids = np.array([]).reshape(n,0) #Initializing centroid randomly
#Here centroid is a n*K dimensional matrix
for i in range(K):
    rand = rd.randint(0,m-1)
    Centroids = np.c_[Centroids,X[rand]]
Output = {} #To store the outputs in a dictionary(cluster number as key and data points as values)
#Finding the min distance and storing the index in vactor C
EuclidianDistance = np.array([]).reshape(m,0)
for k in range(K):
    tempDist = np.sum((X-Centroids[:,k])**2,axis=1)
    EuclidianDistance = np.c_[EuclidianDistance,tempDist]
C = np.argmin(EuclidianDistance,axis = 1) + 1

Y = {} #Temporary Dictionary for storing solution for a particular iteration
for k in range(K):
    Y[k+1] = np.array([]).reshape(2,0)
for i in range(m):
    Y[C[i]] = np.c_[Y[C[i]],X[i]]
     
for k in range(K):
    Y[k+1] = Y[k+1].T
    
for k in range(K):
     Centroids[:,k] = np.mean(Y[k+1],axis=0)
for i in range(n_iter):
     #Step 2.a is repeated till convergence is achieved
      EuclidianDistance=np.array([]).reshape(m,0)
      for k in range(K):
          tempDist=np.sum((X-Centroids[:,k])**2,axis=1)
          EuclidianDistance=np.c_[EuclidianDistance,tempDist]
      C=np.argmin(EuclidianDistance,axis=1)+1
     #step 2.b is repeated
      Y={}
      for k in range(K):
          Y[k+1]=np.array([]).reshape(2,0)
      for i in range(m):
          Y[C[i]]=np.c_[Y[C[i]],X[i]]
     
      for k in range(K):
          Y[k+1]=Y[k+1].T
    
      for k in range(K):
          Centroids[:,k]=np.mean(Y[k+1],axis=0)
      Output=Y

#Visualizing the algorithm and original unclustered data
plt.scatter(X[:,0],X[:,1],c='black',label='unclustered data')
plt.xlabel('eruptions')
plt.ylabel('waiting')
plt.legend()
plt.title('Plot of data points')
plt.show()

#Plotting the clustered Data
color=['red','blue','green']
labels=['cluster1','cluster2','cluster3']
for k in range(K):
    plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
plt.scatter(Centroids[0,:],Centroids[1,:],s=300,c='black',label='Centroids')
plt.xlabel('eruptions')
plt.ylabel('waiting')
plt.legend()
plt.show()
