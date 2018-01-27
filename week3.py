# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
#create a series with an arbitrary list
s= pd.Series([7, 'Heisenberg', 3.14, -1789405969, 'Happy Ending!'])
s
print(s[4])

# specify an index to use when creating the Series
s = pd.Series([7, 'Heisenberg', 3.14, -1789710578, 'Happy Eating!'],index=['A', 'Z', 'C', 'Y', 'E'])
s
s[0] # Access using index location
s['Z'] # Access using label


#from dictionaries to series
d = {'Chicago': 1000, 'New York': 1300, 'Portland': 900, 'San Francisco': 1100,'Austin': 450, 'Boston': None}
cities = pd.Series(d)
cities
print (cities.isnull())

data = {'year': [2010, 2011, 2012, 2011, 2012, 2010, 2011, 2012], 'team': ['Bears', 'Bears', 'Bears', 'Packers', 'Packers', 'Lions', 'Lions', 'Lions'],'wins': [11, 8, 10, 15, 11, 6, 10, 4],'losses': [5, 8, 6, 1, 5, 10, 6, 12]}
football = pd.DataFrame(data, columns=['year', 'team', 'wins', 'losses'])
print ("Access by index ver 1", football['losses'])
print ("Access by index ver 2", football.losses)
print(football)

#Now plot with pandas
# import matplotlib
import matplotlib.pyplot as plt
plt.scatter(football.losses, football.wins)


# modify the data
football['wins'] = 7
print (football)

# you can easily get basic information on your data
print ("Minimum wins: ", football['wins'].min()) # gets you the minimum of a column
# for a quick look at summary statistics, the following is helpful
print(football.describe())
# wen you run this, notice that "year" column is not listed. This is due to data types which you can check by
print(football.dtypes)

football[2:5] # This will get you the rows from 3 to 5
football[['year', 'wins']] # another way (not slicing) to get multiple columns
# why double brackets?
        
football.iloc[[3, 1]] # or this
football['year'][2:5] # This will get you the rows 3 to 5 but only for the year column

football[football.year > 2011] # gets you only the information 2012 and onwards
# can build even more complex queries by combining using logical operators, | (OR), & (AND)
football[(football.wins > 5) & (football.losses < 5)] # gets you those teams with less than 5 losses and more than 5 wins        
        

# now create two data frames

left_frame = pd.DataFrame({'key': range(5),'left_value': ['a', 'b', 'c', 'd', 'e']})
right_frame = pd.DataFrame({'key': range(2, 7),'right_value': ['f', 'g', 'h', 'i', 'j']})
left_frame
right_frame


# now merge these data frames based on the 'key' field -- a common task when merging data sources
print ("--- An inner join:")
print (pd.merge(left_frame, right_frame, on='key'))
# Now add a new element to the left_frame with a key value 2 and assign it to a new variable
left_frame_new = left_frame.append([dict(key=2, left_value='dd')])
# and now try merging the new array
print (pd.merge(left_frame_new, right_frame, on='key'))
# different types of merging are possible
print ("--- Now a left outer join:")
print (pd.merge(left_frame, right_frame, on='key', how='left')) # notice the NAN values - reflect on why they exist
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html
# .merge how argument


#this is to open the File in Python
#f = open('passengerData.csv') 
#for row in csv.reader(f):
    #print(row)

#This is to read the cvs file
passenger_df = pd.read_csv('passengerData.csv')
print(passenger_df)

#This is to read the excel file
ticket= pd.ExcelFile('ticketPrices.xlsx')
ticket
# View the excel file's sheet names
ticket.sheet_names
# Load the xls file's Sheet1 as a dataframe
ticket_df = ticket.parse('Sheet1')
ticket_df

#merge the two files based on ticketype
merged_df= (pd.merge(passenger_df, ticket_df, on='TicketType'))
print (merged_df)

#sort dataframe based on age to display oldest passengers
passenger_df.sort_values(by= ['Age'], ascending= False)

#Plot the data on a scatter plot that shows the Age vs. Ticket Prices
import matplotlib.pyplot as plt
age= merged_df['Age']
price= merged_df['Fare']
print(age)
print(price)
plt.scatter(age, price)

#Plot only the data that shows female passengers aged 
#40 to 50 and who paid more than or equal to 40.
rich_milf= merged_df[(merged_df.Sex=='female') & (merged_df.Age > 39) & (merged_df.Age < 51) & (merged_df.Fare >= 40)] 
print(rich_milf)



####PART2. MISSING VALUES

import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f', 'h'],columns=['one', 'two', 'three'])
print(df)
df['four'] = 'bar'
print(df)
df['five'] = df['one'] > 0
print(df)

# Here we are generating some missing values artificially 
df2 = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
print(df2)

# use this function to get an overview of the nul values along a column
pd.isnull(df2['one'])


#2.1.	Load the slightly modified Titanic survival data into a pandas data frame.
titanic_df = pd.read_csv('titanicSurvival_m.csv')
print(titanic_df)

#2.2. Find the counts of missing values in each column
count_missing= titanic_df.isnull().sum(axis=0)
print(count_missing)

#3.	Compute the mean and other descriptive statistics with describe()
titanic_df.describe()

#2.4.29.	Replace the missing values in "Age" and "Fare" columns with 0 values, and visualise in a scatterplot

import matplotlib.pyplot as plt

age_fill0= titanic_df['Age'].fillna(0)
print(age_fill0)
price_fill0= titanic_df['Fare'].fillna(0)
print(price_fill0)
plt.scatter(age_fill0, price_fill0)

#mean
age_fillm= titanic_df['Age'].fillna(titanic_df['Age'].mean())
print(age_fillm)
price_fillm= titanic_df['Fare'].fillna(titanic_df['Fare'].mean())
print(price_fillm)
plt.scatter(age_fillm, price_fillm)


####EXERCISE 3
#DATA FROM TUBERCOLOSIS DATASET (week1)
#3.1 open csv file as pd dataset
tbc_df = pd.read_csv('TB_burden_countries_2014-09-29.csv')
print(tbc_df)

#count missing vaues in a column
tbc_df['e_prev_100k_hi'].isnull().sum()
tbc_df['e_tbhiv_prct_lo'].isnull().sum()

#alternative function to get a sum of missing data
lista= tbc_df['e_tbhiv_prct_lo']
print (lista)
nanlist=[]
for i in range(len(lista)):
    if np.isnan(lista[i]):
        nanlist.append(i)
print(len(nanlist))       

nanlist=[]
for i in range(len(e_prev_100k_hi_NOMISS)):
    if np.isnan(e_prev_100k_hi_NOMISS[i]):
        nanlist.append(i)
print(nanlist) 


#replace all missing values with 0

#replace missing values of the variables I will use
d1 = tbc_df
d2 = tbc_df
d1.fillna(0)
d2['e_prev_100k_hi'].fillna(d2['e_prev_100k_hi'].mean())
d2['e_inc_100k_lo'].fillna(d2['e_inc_100k_lo'].mean())
d2['e_inc_tbhiv_num_hi'].fillna(d2['e_inc_tbhiv_num_hi'].mean())



e_prev_100k_hi_NOMISS= tbc_df['e_prev_100k_hi'].fillna(tbc_df['e_prev_100k_hi'].mean())
e_inc_100k_lo_NOMISS= tbc_df['e_inc_100k_lo'].fillna(tbc_df['e_inc_100k_lo'].mean())
e_inc_tbhiv_num_hi_NOMISS= tbc_df['e_inc_tbhiv_num_hi'].fillna(tbc_df['e_inc_tbhiv_num_hi'].mean())
#check
print(e_inc_100k_lo_NOMISS)
e_inc_100k_lo_NOMISS.isnull().sum()


#what's the function in R to get the mean value of every column in the dataframe, like colMeans in R
#http://hamelg.blogspot.co.uk/2015/11/python-for-data-analysis-part-16.html
#colmeans = tbc_df.sum()/tbc_df.shape[0]
#TypeError: unsupported operand type(s) for /: 'str' and 'int'
#avg and slicing of the column

#histogram of some columns
import matplotlib.pyplot as plt
plt.hist(d1, 50, normed=1, facecolor='blue', alpha=0.75)
plt.hist(d2, 50, normed=1, facecolor='blue', alpha=0.75)

plt.hist(e_prev_100k_hi_NOMISS, 50, normed=1, facecolor='blue', alpha=0.75)
plt.hist(e_inc_100k_lo_NOMISS, 50, normed=1, facecolor='green', alpha=0.75)
plt.hist(e_inc_tbhiv_num_hi_NOMISS, 75, normed=1, facecolor='green', alpha=0.75)

#log transformation + visualization
import numpy as np
plt.hist(np.log(e_prev_100k_hi_NOMISS), 50, normed=1, facecolor='blue', alpha=0.75)
### problem ValueError: range parameter must be finite.
plt.hist(np.log(e_inc_100k_lo_NOMISS), 50, normed=1, facecolor='green', alpha=0.75, range=(e_inc_100k_lo_NOMISS.min(),e_inc_100k_lo_NOMISS.max()))
plt.hist(np.log(e_inc_tbhiv_num_hi_NOMISS), 50, normed=1, facecolor='red', alpha=0.75)

#choose only the numerical column
#https://stackoverflow.com/questions/25039626/how-do-i-find-numeric-columns-in-pandas
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.select_dtypes.html
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
newdf = tbc_df.select_dtypes(include=numerics)
print(newdf)

#map all the column to [0,1] interval
#just one column...
#https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
a=newdf['e_mort_exc_tbhiv_100k']
print(a)
norm_a= (a-min(a))/(max(a)-min(a))
print(norm_a)
plt.hist(norm_a, 50, normed=1, facecolor='blue', alpha=0.75)


#https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

norm_newdf=normalize(newdf)
plt.hist(norm_newdf['e_inc_num'], 50, normed=1, facecolor='blue', alpha=0.75)
#error ValueError: max must be larger than min in range parameter.



