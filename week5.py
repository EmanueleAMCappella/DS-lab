# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:16:27 2017

@author: EAMC
"""
import csv as csv 
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

census = pd.read_csv('censusCrimeClean.csv')
print (census.head()) # check if everything is in place

numpyArray = census.as_matrix(["medIncome", "ViolentCrimesPerPop"])

print ("Shape: ", numpyArray.shape)

arr1 = numpyArray[:,0] # now this holds "medIncome" values
arr2 = numpyArray[:,1] # this holds "ViolentCrimesPerPop" values

#%%
######################
#Correlation analysis #
###################### 

# First do a Pearson correlation
corrPearson, pVal1 = stats.pearsonr(arr1, arr2)
print ("Correlation Pearson: ", corrPearson, pVal1)

# followed by a Spearman
corrSpearman, pVal2 = stats.spearmanr(arr1, arr2)
print ("Correlation Spearman: ", corrSpearman, pVal2)

# now on to drawing
plt.figure(1)
plt.suptitle('medIncome vs. ViolentCrimesPerPop')
plt.xlabel('medIncome')
plt.ylabel('ViolentCrimesPerPop')
plt.scatter(arr1, arr2 , c = "#D06B36", s = 50, alpha = 0.4, linewidth='0')


#%%
######################
#Regression Analysis #
######################    

# We first use the linear regression from scipy library which
# turns 4 values. We save all the results in variables
# Notice that in addition to the slope, we also get the intercept of the line
slope, intercept, r_value, p_value, std_err = stats.linregress(arr1, arr2)

print ("Slope: ", slope)
print ("Intercept: ", intercept)
print ("p_value: ", p_value)
print ("std_err: ", std_err)

# It is always a good idea to visualise the model together with the data 
# we can use the polyval function to evaluate our model over the whole set of data points
# What this does is to simply run y = m * x + b function for all our points 
plt.figure(2)
plt.suptitle('medIncome vs. ViolentCrimesPerPop')
plt.xlabel('medIncome')
plt.ylabel('ViolentCrimesPerPop')
plt.scatter(arr1, arr2 , c = "#D06B36", s = 50, alpha = 0.4, linewidth='0')

xp = np.linspace(arr1.min(), arr1.max(), 100)
evaluatedLine = np.polyval([slope, intercept], xp)
# let's see a black line overlaid on the data
plt.plot(xp, evaluatedLine, 'k--', linewidth = 3)


# BONUS:
# We notice that the relation might be better captured with a higher order polynomial
# so we can try to fit a second order curve
pCoeff = np.polyfit(arr1, arr2, 2)
evaluatedCurve = np.polyval(pCoeff, arr1)
           
xp = np.linspace(arr1.min(), arr1.max(), 100)
polynomial = np.poly1d(pCoeff)
plt.plot(xp, polynomial(xp), linewidth = 3)

#We can see that the second order polynomial captures the nature of the relation slightly better. 


#%%
#######################
# MULTIPLE REGRESSION #
#######################

# import statsmodels
import statsmodels.api as sm

# this time get two independent variables, i.e., regressors 
regressors = census.as_matrix(["medIncome", "PctTeen2Par"])

# our dependent variable is still "ViolentCrimesPerPop" which we still keep in variable arr2
# we use the OLS function from statsmodels
model = sm.OLS(arr2,regressors)
results = model.fit()

# the summary function returns a very comprehensive report on the results
print ("Params: ", results.summary())

# This method uses R style patsy formulas to indicate the relations
# Have a look at the patsy documentation if you like to read more.
# But within the scope of this course we will not go into its details
model2 = sm.formula.ols(formula='ViolentCrimesPerPop ~ medIncome + PctTeen2Par', data=census, hasconst = False)
results2 = model2.fit()

print ("Params 2: ", results2.summary())
print ("---------------------------------------------") 

# In order to get the same result without using formulas, one can add a constant column.
# statsmodels has a special function to add constants: 
regressors = sm.add_constant(regressors)

# These two following options give a slightly different R^2 although everything else in the summary result is the same.
# my guess is that there is a slight bug here with statsmodels in the way that R^2 is computed. 
#For the sake of compatibility with the above, I would leave this as : hasconst = False

#model3 = sm.OLS(arr2,regressors)
model3 = sm.OLS(arr2,regressors, hasconst = False)
results3 = model3.fit()

print ("Params 3: ", results3.summary())
print ("---------------------------------------------")