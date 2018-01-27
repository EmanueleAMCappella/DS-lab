# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:05:47 2017

@author: EAMC
"""

import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#read csv file as pandas dataframe
prop_car = pd.read_csv('accord_sedan.csv')
print(prop_car)

#Visualise the columns: "price"	and "mileage" + 3.Identify the 2D outliers using the visualisation
print(prop_car.price)
print(prop_car.mileage)

plt.xlabel('price')
plt.suptitle('Price vs. Mileage')
plt.ylabel('mileage')
plt.scatter(prop_car.price, prop_car.mileage , linewidth = 0)

#%%

# Add two new columns to the dataframe called isOutlierPrice, isOutlierMileage. 
#For the price column, calculate the mean and standard deviation. 
#Find any rows that are more than 2 times standard deviations away from the mean 
#and mark them with a 1 in the isOutlierPrice column. Do the same for mileage column

#https://stackoverflow.com/questions/16327055/how-to-add-an-empty-column-to-a-dataframe
#create two new columns of Nan Values
new_v=prop_car.reindex(columns=['isOutlierPrice', 'isOutlierMileage'])
print(new_v)

#append them to the dataset
prop_car=prop_car.join(new_v)
print(prop_car)

##SIMPLER WAY TO CREATE A 000 column...
prop_car["isOutlierPrice"] = 0 
prop_car["isOutlierMileage"] = 0


#compute mean and std for the two variables
avg_price=prop_car.price.mean()
avg_mileage= prop_car.mileage.mean()

sd_price= prop_car.price.std()
sd_mileage= prop_car.mileage.std()

#compute 2std variable
sd_price_2 = sd_price*2
sd_mileage_2= sd_mileage*2


#Find any rows that are more than 2 times standard deviations away from the mean 
#and mark them with a 1 in the isOutlierPrice column. Do the same for mileage column

#If you run the code below there will be an error
#SettingWithCopyWarning was created to flag potentially confusing "chained" assignments, 
#such as the following, which don't always work as expected, 
#particularly when the first selection returns a copy.
#if you run first the code below, the error message does not appear
#taken from https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None

for i in prop_car['price']:
    if abs(i - avg_price) > sd_price_2:
        prop_car["isOutlierPrice"][i]= 1
    else:
        prop_car["isOutlierPrice"][i]= 0

#Better approach from solutions. abs() is to get absolute values
for i, val in enumerate(prop_car['price']):
    if abs(val - avg_price) > sd_price_2:
        prop_car["isOutlierPrice"][i] = 1

# The above can be done much more efficiently and elegantly using numpy constructs:
# we are using the "where" operator here and the conditions are run on all the array elements efficiently
# notice that we need to use "bitwise operators" to implement AND, OR, NOT operations here.

prop_car['isOutlierPrice'] = np.where(abs(prop_car['price'] - avg_price) > sd_price_2, 1, 0)

#same thing for mileage
for i, val in enumerate(prop_car['mileage']):
    if abs(val - avg_mileage) > sd_mileage_2:
        prop_car["isOutlierMileage"][i] = 1  

#or...       
prop_car['isOutlierMileage'] = np.where(abs(prop_car['mileage'] - avg_mileage) > sd_mileage_2, 1, 0)

print (prop_car.mileage)
print(prop_car.isOutlierMileage)
#Visualise these values with a different color in the plot. 
#Observe whether they are the same as you would mark them.

# This part helps us to generate a color array with different colors for the 1D outliers we compute
# first create an empty list
colorColumn = []
# we make use of the HEX color codes to use nicely distinguisable colors
# Adobe Color CC is a nice resource to find colors (https://color.adobe.com/)
for i in range(len(prop_car)):
    if prop_car["isOutlierPrice"][i] == 1:
        colorColumn.append("#207F24") # green
    elif prop_car["isOutlierMileage"][i] == 1:
        colorColumn.append("#4C18BF") # a blueish color
    else:
        colorColumn.append("#B9BCC0") # gray

## One thing to notice in the above process is our decision to mark data points with a hard threshold. i.e., two standard deviations away from the centre, 
#we would recommend you to try out with higher/lower values of this threshold (or even use versions based on IQR) and see how much it changes your observations. 
#It always makes sense to visualise what is marked as an outlier and evaluate them as the analyst yourself.
        
plt.figure(2)
plt.xlabel('price')
plt.suptitle('Price vs. Mileage')
plt.ylabel('mileage')
plt.scatter(prop_car.price, prop_car.mileage , c = colorColumn, s = 50, linewidth='0')

#%%

#We notice that our 1D outlier measure has selected points that are on the outer edges of the distribution. 
#There are points that are outliers according to one and not according to the other. There are some that might be 2D outliers. 
#In order to detect that, we need high-dimensional methods. 
#Mahalanobis distance is one such measure. We follow with this optional exercise:

# We now get a part of the data frame as a numpy matrix to use in scipy
columnValues = prop_car.as_matrix(["price", "mileage"])

# In order to generate a "mean vector", we use the mean values already computed above.
# Notice that we make use of the reshape() function to get the mean vector in a compatible shape
# as the data values.
meanVector = np.asarray([avg_price, avg_mileage]).reshape(1,2)

# We make us of the scipy function which does the computations itself.
# Alternatively, one can provide a covariance matrix that is computed outside as a parameter.
# In cases where robustness of the covariance matrix is the issue, this can be a good option.

# first import the spatial subpackage from scipy
from scipy import spatial

mahalanobisDistances = spatial.distance.cdist(columnValues, meanVector, 'mahalanobis')[:,0]

# We create a new figure where we use a color mapping and use the computed mahalanobis distances 
# as the mapping value
plt.figure(3)
plt.xlabel('price')
plt.suptitle('Price vs. Mileage')
plt.ylabel('mileage')
plt.scatter(prop_car.price, prop_car.mileage , c = mahalanobisDistances, cmap = plt.cm.Greens, s = 50, linewidth= 0)

#We can see that more "central" points have a low Mahalanobis distance while those on the outskirts have more saturated colors, i.e., higher distance to the centre. 
#In this plot, the only clear 2D outlier is the dark point on the left top corner. 
#Notice that Mahalanobis distance takes the variation of the data into account and it is very powerful to capture elliptical relations as it makes use of the covariance information. In cases where you have several more variables (and it is hard to visually determine outliers), Mahalanobis distance becomes very handy.


#%%
##################
####### QQ-PLOTS #
##################

#With the code below, we first choose a column and investigate its shape through a Q-Q plot. 
#In order to observe the different shapes better and in particular to observe how data transformations affect the data, 
#we'll try here to apply log() to our data.

tbDF= pd.read_csv('TB_burden_countries_2014-09-29.csv')

# Before we proceed fill all missing values
tbDF = tbDF.fillna(tbDF.mean())

# Let's choose columns here:
columnToTest = "e_prev_100k_hi" #"e_inc_tbhiv_num"#"c_cdr" #"e_prev_num" #"e_prev_100k_hi"

# You can also investigate the shape of this column, which is not really normal but less skewed
#columnToTest = "c_cdr"

plt.style.use('ggplot')
# And we make use of the qqplot function from the stasmodel package
import scipy.stats as stats

fig = sm.qqplot(tbDF[columnToTest], stats.norm, line = 'r')
plt.show()

plt.suptitle(columnToTest)
plt.hist(tbDF[columnToTest], 20)

# This column is highly skewed, let's observe how a log() operation 
#will affect the shape of the distribution and how the Q-Q plot can reveal that.

# We'll use the np.log() function and use it as a parameter to the pandas .apply() function
fig = sm.qqplot(tbDF[columnToTest].apply(np.log), stats.norm, line = 'r')
plt.suptitle("---- This time with log() of the column ----")
plt.show()

plt.suptitle("log(" + columnToTest + ")")
plt.hist(tbDF[columnToTest].apply(np.log), 20)


#%%
####################
#Robust statistics #
####################

#In cases where the data is likely to have more outliers and the estimations thus less reliable, robust statistics are of great help as they tend to be more resistent against outliers and also the more "non-standard" shapes of distributions. 
#In the following, we observe how different the conventional and robust versions of descriptive statistics can differ:

# Here we choose a skewed data column from the data and observe the 
colToTest = "e_prev_100k_hi" 
plt.boxplot(tbDF[colToTest])

print('mean:', tbDF[colToTest].mean(),'median', tbDF[colToTest].median())

print('std:', tbDF[colToTest].std(),'iqr', stats.iqr(tbDF[colToTest]))

print('std:', tbDF[colToTest].std(),'mad', sm.robust.scale.mad(tbDF[colToTest]))

