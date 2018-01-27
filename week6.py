# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:17:04 2017

@author: EAMC
"""

################################
# DIMENSION REDUCTION WITH PCA #
################################

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

census = pd.read_csv('censusCrimeClean.csv')
print (census.head()) # check if everything is in place

columnNames = np.asarray(census.columns.values)

numpyArray = census.as_matrix()
arrayForPCA = numpyArray[:,1::]
columnNamesFiltered = columnNames[1::]

# Build a model that will return two principal components
pca = PCA(n_components=2)

# We first fit a PCA model to the data
pca.fit(arrayForPCA)

# First have a look at the component loadings. 

# have a look at the components directly if we can notice any interesting structure
projectedAxes = pca.transform(arrayForPCA)

# now on to drawing
plt.figure(1)
plt.suptitle('First two components')
plt.xlabel('PC_1')
plt.ylabel('PC_2')
plt.scatter(projectedAxes[:,0], projectedAxes[:,1], c = "#D06B36", s = 50, alpha = 0.4, linewidth='0')
                        
# some strong patterns in PCA results, why is that?
# let's have a look at the component loadings to identify this

print( pca.components_)

# this is lots of numbers, right? So here we face a challenge
# it is not easy to interpret these when we have lots of columns in the computations.
# Since we have the strong structures in PC_1, we focus only on the 1st component first.

# We do a couple of operations here:
# Look at only 1st component: pca.components_[0]
# We are interested in the magnitude of the loadings, i.e., we need absolute numbers: np.abs(pca.components_[0])
# We would like to get the names of the columns to make some sense
# so we order the loadings from high-to-low and look at the names (argsort gives the order of the IDs)
# let's look at the first 10 only

comp1Loadings = np.asarray(pca.components_[0])[np.argsort( np.abs(pca.components_[0]))[::-1]][0:10]
comp1Names = np.asarray(columnNamesFiltered)[np.argsort( np.abs(pca.components_[0]))[::-1]][0:10]

for i in range(0, 10):
    print ( "Column \"" , comp1Names[i] , "\" has a loading of: ", comp1Loadings[i])            
            
#And we notice that the column called "fold" has a very high loading, almost 1. And this is a sign that we need to have a look at this column. 
#When we look at the excel sheet, we can see that it is an artefact in the data (which was artificially added for cross-correlation purposes). 
            
# let's get the fold column out and run the analysis again:

arrayForPCA = numpyArray[:,2::]
# keep the filtered column names
columnNamesFiltered = columnNames[2::]

# Build a model that will return two principal components
pca = PCA(n_components=2)

# We first fit a PCA model to the data
pca.fit(arrayForPCA)

# have a look at the components directly if we can notice any interesting structure
projectedAxes = pca.transform(arrayForPCA)

# as an additional task, we think it might be nice to see the crime volume as mapped to the color of the points
colorMappingValuesCrime = np.asarray(arrayForPCA[:,-1], 'f')

# now on to drawing
plt.figure(2)
plt.suptitle('PCs for US Census & Crime with "fold" removed')
plt.xlabel('PC_1')
plt.ylabel('PC_2')
plt.scatter(projectedAxes[:,0], projectedAxes[:,1], c = colorMappingValuesCrime, cmap = plt.cm.Greens, s = 50, linewidth='0')
# Here I say I would like to look at the first ten columns
numberOfColumnsToLook = 10

print ("--- Firstly, the first component: ")
comp1Loadings = np.asarray(pca.components_[0])[np.argsort( np.abs(pca.components_[0]))[::-1]][0:numberOfColumnsToLook]
comp1Names = np.asarray(columnNamesFiltered)[np.argsort( np.abs(pca.components_[0]))[::-1]][0:numberOfColumnsToLook]

for i in range(0, numberOfColumnsToLook):
    print ( "Column \"" , comp1Names[i] , "\" has a loading of: ", comp1Loadings[i])
    
print ("\n --- Secondly, the second component: ")
comp2Loadings = np.asarray(pca.components_[1])[np.argsort( np.abs(pca.components_[1]))[::-1]][0:numberOfColumnsToLook]
comp2Names = np.asarray(columnNamesFiltered)[np.argsort( np.abs(pca.components_[1]))[::-1]][0:numberOfColumnsToLook]

for i in range(0, numberOfColumnsToLook):
    print ( "Column \"" , comp2Names[i] , "\" has a loading of: ", comp2Loadings[i])            
            
 #What we do is to dig deeper within the first component and see whether we can find any inter relations 
 #between the 10 income related columns listed above. Notice that we are now reducing a 10-dimensional data set.

# a local analysis
    
columnsToAnalyse = np.argsort( np.abs(pca.components_[0]))[::-1][0:numberOfColumnsToLook]
columnNamesFiltered = columnNamesFiltered[columnsToAnalyse]
filteredLocalArray = arrayForPCA[:,columnsToAnalyse]

# Build a model that will return two principal components
pca3 = PCA(n_components=2)

# We first fit a PCA model to the data
pca3.fit(filteredLocalArray)

# have a look at the components directly if we can notice any interesting structure
projectedAxes = pca3.transform(filteredLocalArray)

# now on to drawing
plt.figure(3)
plt.suptitle('PCs for a sub group of income related variables')
plt.xlabel('PC_1')
plt.ylabel('PC_2')
plt.scatter(projectedAxes[:,0], projectedAxes[:,1], c = colorMappingValuesCrime, cmap = plt.cm.Greens, s = 50, linewidth='0')


print ("A local analysis")
print ("--- first component: ")
comp1Loadings = np.asarray(pca3.components_[0])[np.argsort( np.abs(pca3.components_[0]))[::-1]][0:numberOfColumnsToLook]
comp1Names = np.asarray(columnNamesFiltered)[np.argsort( np.abs(pca3.components_[0]))[::-1]][0:numberOfColumnsToLook]

for i in range(0, numberOfColumnsToLook):
    print ( "Column \"" , comp1Names[i] , "\" has a loading of: ", comp1Loadings[i])
    
print ("\n --- second component: ")
comp2Loadings = np.asarray(pca3.components_[1])[np.argsort( np.abs(pca3.components_[1]))[::-1]][0:numberOfColumnsToLook]
comp2Names = np.asarray(columnNamesFiltered)[np.argsort( np.abs(pca3.components_[1]))[::-1]][0:numberOfColumnsToLook]

for i in range(0, numberOfColumnsToLook):
    print ( "Column \"" , comp2Names[i] , "\" has a loading of: ", comp2Loadings[i])

#%% ATTENTION:BUG
####################################
# MULTIDIMENSIONAL SCALING #########
####################################

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn import manifold
from sklearn.metrics import euclidean_distances

londonBorough = pd.read_excel('london-borough-profiles.xlsx')

# since the data is problematic and the missing values were 
# represented with all sorts of strange characters, we can do some tricks
# and try to force everything to be converted into numeric arrays
# this will find the suitable data type and get the problematic values filled with NAN

# since the data is problematic and the missing values were 
# represented with all sorts of strange characters, we can do some tricks
# and try to force everything to be converted into numeric arrays
# this will find the suitable data type and get the problematic values filled with NAN

londonBorough = londonBorough.convert_objects(convert_numeric=True)

# and let's get only the numeric columns
numericColumns = londonBorough._get_numeric_data()

# keep place names and store them in a 
placeNames = londonBorough["Area/INDICATOR"]

# let's fill the missing values with mean()
numericColumns = numericColumns.fillna(numericColumns.mean())

# let's centralize the data
numericColumns -= numericColumns.mean()

# now we compute the euclidean distances between the columns by passing the same data twice
# the resulting data matrix now has the pairwise distances between the boroughs.
# CAUTION: note that we are now building a distance matrix in a high-dimensional data space
# remember the Curse of Dimensionality -- we need to be cautious with the distance values
distMatrix = euclidean_distances(numericColumns, numericColumns)

# for instance, typing distMatrix.shape on the console gives:
# Out[115]: (38, 38) # i.e., the number of rows

# first we generate an MDS object which returns
mds = manifold.MDS(n_components = 2, max_iter=3000, n_init=1, dissimilarity="precomputed")
Y = mds.fit_transform(distMatrix)


fig, ax = plt.subplots()
plt.suptitle('MDS on only London boroughs')
ax.scatter(Y[:, 0], Y[:, 1], c="#D06B36", s = 100, alpha = 0.8, linewidth='0')

for i, txt in enumerate(placeNames):
    ax.annotate(txt, (Y[:, 0][i],Y[:, 1][i]))

# get the data columns relating to emotions and feelings
dataOnEmotions = numericColumns[["Life satisfaction score 2012-13 (out of 10)", "Worthwhileness score 2012-13 (out of 10)","Happiness score 2012-13 (out of 10)","Anxiety score 2012-13 (out of 10)"]]

# a new distance matrix to represent "emotional distance"s
distMatrix2 = euclidean_distances(dataOnEmotions, dataOnEmotions)

# compute a new "embedding" (machine learners' word for projection)
Y2 = mds.fit_transform(distMatrix2)

# let's look at the results
fig, ax = plt.subplots()
plt.suptitle('An \"emotional\" look at London boroughs')
ax.scatter(Y2[:, 0], Y2[:, 1], c="#D06B36", s = 100, alpha = 0.8, linewidth='0')

for i, txt in enumerate(placeNames):
    ax.annotate(txt, (Y2[:, 0][i],Y2[:, 1][i]))






            
