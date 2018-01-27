# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

##PART 1

#for row in csv.reader(f):
    #print(row)

import csv     # imports the csv module
import sys      # imports the sys module

#check your current directory and the files location
import os
cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in '%s': %s" % (cwd, files))

#%%
############################### 
#EXERCISE 1. Counting columns #
###############################
  
f = open('TB_burden_countries_2014-09-29.csv', encoding='utf-8') # opens the csv file
rowCount = 0 # declare a variable to keep the row count value
for row in csv.reader(f):
    rowCount += 1
print("There are ", rowCount , "columns in this dataset.")

#%% 
#####################################
##  EXERCISE 2. Finding the averages#
#####################################

# A very crude way of getting averages along a column
# but hang on, we'll be doing more efficient operations soon

rowCount = 0 # declare a variable to keep the row count value
totalValue = 0 # declare a variable to keep the column total

f = open('TB_burden_countries_2014-09-29.csv') # opens the csv file

l = next(f) # This comment skips the column headers since we don't want them in our averaging. We might do something with the header line if we wanted to.
colHeaders = l.split(',') # we split the col names string since they are comma separated.
#print (colHeaders)

# This is not really needed but makes it easier to find the column names
for i, colName in enumerate(colHeaders):
    print ("Column: ", i, " is ",colName)

# Let's compute the average for e_prev_num_lo which has a column id 11.
# notice that although this is the 12th column, we'll get its values by referencing to the 11th.
# computers count from 0 unlike us humans !!
i = 0
for row in csv.reader(f):
    rowCount += 1    
    i+=1
    # We need to be careful here since we need to convert the csv data which are originally strings.
    # Most of the time there are problems in the data, such as gaps, missing values, etc.
    # If you look at the spreadsheet, you'll notice some empty cells.
    # The try-except struct makes sure that those empty cells are skipped and our code don't crash.
    try:
        val = float(row[11])
        totalValue += val
    except ValueError:
        pass
    
# Here we're checking to make sure that we are trying to divide by zero.
if rowCount > 0:
    columnAverage = totalValue / rowCount    
    print("Column Average:", columnAverage)
    
#%%
###########################################################
#EXERCISE 3. Finding the averages for two different years #
###########################################################
    
rowCount1990 = 0 # declare a variable to keep the row count value for the rows dated 1990
total1990 = 0 # declare a variable to keep the total number of rows in 1990

rowCount2011 = 0 # declare a variable to keep the row count value for the rows dated 2011
total2011 = 0  # declare a variable to keep the total number of rows in 2011

f = open('TB_burden_countries_2014-09-29.csv') # opens the csv file

l = next(f) # This comment skips the column headers since we don't want them in our averaging. We might do something with the header line if we wanted to.
colHeaders = l.split(',') # we split the col names string since they are comma separated.


#  e_prev_num_lo is column id 11, year is column id 5

i = 0
for row in csv.reader(f):
    rowCount += 1    
    i+=1    
    
    try:
        year = int(row[5])        
    except ValueError:
        continue 
        # this makes sure that if there is no year (i.e., missing), we stop this iteration 
        #and move on the the next line, so the code below don't run in this case  
    
    try:
        val = float(row[11])        
    except ValueError:
        continue
    # Here we use an if clause to make sure that we are using the correct years.
    if year == 1990:
        total1990 += val
        rowCount1990 += 1
        
    elif year == 2011:
        total2011 += val
        rowCount2011 += 1

if rowCount1990 > 0:
    columnAverage = total1990 / rowCount1990
    print("Average for 1990:", columnAverage, " with ", rowCount1990, " data points."  )

if rowCount2011 > 0:
    columnAverage = total2011 / rowCount2011
    print("Average for 2011:", columnAverage, " with ", rowCount2011, " data points."  )
    
#%%  

##########################################    
####PART 2: numpy introductory exercises #
##########################################

import numpy as np

#Create an array of int ranging from 5-15
array1 = np.arange(5,16) # notice that we extend the range until 16 here.
print (array1)

#Create an array containing 7 evenly spaced numbers between 0 and 23:
array2 = np.linspace(0, 23, num = 7)
print(array2)

# generate data
vals1 = np.random.uniform(-1, 1, 100) # Here, we are drawing n=100 samples from a uniform dist. 
#Try altering the number and observe the shape of the resulting histogram using the code that follows.
#Visualise the array in an histogram in matplotlib:
import matplotlib.pyplot as plt

plt.hist(vals1, 10) #We can change the number of bins, what is a correct value?
plt.title("Uniform Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()  

#Now try the above by drawing more/less samples from the uniform distribution. Can you still observe the same uniform pattern? Or does it look different? 
#And also try modifying the number of bins in the histogram while keeping the sample size fixed and observe how this changes things. Both of these questions are interesting decisions you need to be careful about while building analysis strategies and visualisations, you might be distorting the patterns or introducing artificial ones through the decisions you make in the analysis process. 

#Create two random numpy arrays with 10 elements. Find the Euclidean distance between the arrays using arithmetic operators, hint: numpy has a sqrt function 

from math import sqrt # works better if you are going to use a single function from a package

arraySize = 10 # put the array size in a variable so that things are flexible
arr1 = np.random.rand(arraySize)
arr2 = np.random.rand(arraySize)

# Now this is the conventional way of doing things.
# We use loops here to iterate over the array and do the arithmetic operations
# one by one.

euclideanDist = 0
for i in range(0, arraySize):
         euclideanDist += (arr1[i] - arr2[i])**2
euclideanDist = sqrt(euclideanDist)

print(euclideanDist)

#Python is not known for its speed when it comes to loops but numpy arrays are built on pretty efficient structures and it is better to do vectorial operations whenever we can and avoid loops. The following line does the same, for instance. Notice how arrays are used like single values -- this is numpy operating on data columns
euclideanDistFast = np.sqrt(np.sum((arr1-arr2)**2))
print (euclideanDistFast)


