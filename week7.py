# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 17:05:30 2017

@author: EAMC
"""

from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
from sklearn import datasets
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

winePD = pd.read_csv('wine.csv')
print (winePD.head()) # check if everything is in place

data = winePD.as_matrix()

# get the known cluster labels into a separate array
# in practical cases, you might not already have such information
# but when you do, these might be used to either evaluate or "supervise"
# the model you are trying to build
classLabelsKnown = np.asarray(data[:,0], 'i') 
classLabelsKnown -= 1 # get the labels start from 0

# get all the data but not the class labels
DataToCluster = data[:,1::]

#computing K-Means with K = 3 (3 clusters)
kmeansModel = KMeans(init='random', n_clusters=3, n_init=10)
kmeansModel.fit_predict(DataToCluster)
clusterResults = kmeansModel.labels_

########################################
## An alternative way  to cluster using Scipy's methods !!!
########################################
# computing K-Means with K = 3 (3 clusters)
#centroids,_ = kmeans(DataToCluster,3)
# assign each sample to a cluster
#idx,_ = vq(DataToCluster,centroids)
#
########################################

## Let's check the results and try to compare with known labels
for i, clustLabel in enumerate(clusterResults):
    print("Cluster result: ", clustLabel, " Known labels: ",classLabelsKnown[i])
    
## This is a problem now!
# We don't really know how to match the labels since the clustering
# algorithm returns arbitrary labels. Each time you run a clustering algorithm, 
# you will be getting different label values as the solutions.
# We observe that we can't really match the IDs

pca = PCA(n_components=2)

# We first fit a PCA model to the data
pca.fit(DataToCluster)

# have a look at the components directly if we can notice any interesting structure
projectedAxes = pca.transform(DataToCluster)

dataColumnsToVisualize = projectedAxes
#dataColumnsToVisualize = data
IDsForvisualization = clusterResults # to color the points according to the results of k-means
#IDsForvisualization = classLabelsKnown # to color the points according to the known labels 

columnIDToVisX = 0 # some variable to keep coind simple and flexible
columnIDToVisY = 1

import matplotlib.pyplot as plt
plt.figure(1)
plt.suptitle('Results of the algorithm visualised over PCs')
plt.xlabel('PC_1')
plt.ylabel('PC_2')
# some plotting using numpy's logical indexing
plt.scatter(dataColumnsToVisualize[IDsForvisualization==0,columnIDToVisX], dataColumnsToVisualize[IDsForvisualization==0,columnIDToVisY], c = "#66c2a5", s = 50, alpha = 0.7, linewidth='0') # greenish
plt.scatter(dataColumnsToVisualize[IDsForvisualization==1,columnIDToVisX], dataColumnsToVisualize[IDsForvisualization==1,columnIDToVisY], c = "#fc8d62", s = 50, alpha = 0.7, linewidth='0') # orangish
plt.scatter(dataColumnsToVisualize[IDsForvisualization==2,columnIDToVisX], dataColumnsToVisualize[IDsForvisualization==2,columnIDToVisY], c = "#8da0cb", s = 50, alpha = 0.7, linewidth='0') # blueish


IDsForvisualization = classLabelsKnown

plt.figure(2)
plt.suptitle('Known labels visualised over PCs')
plt.xlabel('PC_1')
plt.ylabel('PC_2')
plt.scatter(dataColumnsToVisualize[IDsForvisualization==0,columnIDToVisX], dataColumnsToVisualize[IDsForvisualization==0,columnIDToVisY], c = "#66c2a5", s = 50, alpha = 0.7, linewidth='0')
plt.scatter(dataColumnsToVisualize[IDsForvisualization==1,columnIDToVisX], dataColumnsToVisualize[IDsForvisualization==1,columnIDToVisY], c = "#fc8d62", s = 50, alpha = 0.7, linewidth='0')
plt.scatter(dataColumnsToVisualize[IDsForvisualization==2,columnIDToVisX], dataColumnsToVisualize[IDsForvisualization==2,columnIDToVisY], c = "#8da0cb", s = 50, alpha = 0.7, linewidth='0')

#%%
#######################
# CROSS-VALIDATION   ##
#######################

import csv as csv 
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import cross_validation

# first start by loading the data
census = pd.read_csv('censusCrimeClean.csv')

numpyArray = census.as_matrix(["medIncome", "ViolentCrimesPerPop"])

#print ("Shape: ", numpyArray.shape)

arr1 = numpyArray[:,0] # now this holds "medIncome" values
arr2 = numpyArray[:,1] # this holds "ViolentCrimesPerPop" values

# We would also like to plot the regression lines to see how they vary
# but first the points to see the context
plt.figure(1)
plt.suptitle('medIncome vs. ViolentCrimesPerPop')
plt.xlabel('medIncome')
plt.ylabel('ViolentCrimesPerPop')
plt.scatter(arr1, arr2 , c = "#D06B36", s = 50, alpha = 0.4, linewidth='0')

numberOfSamples = len(arr1)

# generate sampling indices for 20 points and set the k to be 5 in this case, 5-fold cross validation
# You can use these to access your data accordingly
    
kf = cross_validation.KFold(numberOfSamples, n_folds=5)
foldCount = 0
for train_index, test_index in kf:
    print("-------- We are having the run: ", foldCount )
    arraySubset1 = arr1[train_index]
    arraySubset2 = arr2[train_index]
    slope, intercept, r_value, p_value, std_err = stats.linregress(arraySubset1, arraySubset2)
    
    print ("Slope: ", slope, "Intercept: ", intercept, "r_value: ", r_value, "p_value: ", p_value,"std_err: ", std_err)
    
    xp = np.linspace(arr1.min(), arr1.max(), 100)
    evaluatedLine = np.polyval([slope, intercept], xp)
    # let's see a black line overlaid on the data
    # for each run, we draw a line again
    plt.plot(xp, evaluatedLine, 'k--', linewidth = 1, alpha = 0.3)
    foldCount += 1 
    
# Alternatively, you can test whether the regression model can estimate unseen points.
foldCount = 0
for train_index, test_index in kf:
    print("-------- We are having the run: ", foldCount )
    arraySubset1 = arr1[train_index]
    arraySubset2 = arr2[train_index]
    
    unseenSubset1 = arr1[test_index]
    unseenSubset2 = arr2[test_index]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(arraySubset1, arraySubset2)
    
    # Use the regression models to estimate the unseen "medIncome" values 
    # given their ViolentCrimesPerPop values.
    estimatedValues = slope * unseenSubset1 + intercept
    
    # check the differences between the estimates and the real values    
    differences = unseenSubset2 - estimatedValues
    print (unseenSubset2)
    print (np.average(differences))
    foldCount += 1 