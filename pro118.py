from numpy.core.fromnumeric import ravel
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import statistics as st
import random as rd
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split as tts 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score as AS
from sklearn.metrics import confusion_matrix as CXM
from sklearn.cluster import KMeans as KM

data_file = pd.read_csv("c118/stars.csv")

data_file_scatter = px.scatter(data_file , x ="Size",y="Light")
# data_file_scatter.show()

X = data_file.iloc[:,[0,1]].values
#  WCSS stands for Within Cluster Sum of Squares
wcss =[]

for i in range(1,11):
    kmeans = KM(n_clusters=i,init="k-means++",random_state=125)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# to plot a line chart to show a elbow like structure
# plt.figure(figsize=(10,5))
# sns.lineplot(range(1,11),wcss,marker="o",color="orange")
# plt.xlabel("number of culster")
# plt.ylabel("wcss")
# plt.title("elbow graph")
# # plt.show()

# to do predection 

kmeans = KM(n_clusters=3,init="k-means++",random_state=125)
Ykmeans = kmeans.fit_predict(X)

plt.figure(figsize=(15,7))
sns.scatterplot(X[Ykmeans==0,0],X[Ykmeans==0,1],color="orange",label="cluster 1")
sns.scatterplot(X[Ykmeans==1,0],X[Ykmeans==1,1],color="teal",label="cluster 2")
sns.scatterplot(X[Ykmeans==2,0],X[Ykmeans==2,1],color="red",label="cluster 3")
# to mark the centroid 
sns.scatterplot(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], color="black", label="centeroid",s =100, marker=",")
plt.xlabel("petal size")
plt.ylabel("sepal size")
plt.title("culster of flower ")
plt.show()