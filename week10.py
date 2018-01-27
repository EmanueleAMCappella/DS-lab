# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 18:12:15 2017

@author: EAMC
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report

from sklearn import preprocessing


url = 'titanicDataFull.csv'
titanic = pd.read_csv(url)
titanic.columns = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
titanic.head()

## We are using Seaborn plots today, they do look more ok
sns.countplot(x='Survived',data=titanic, palette='hls')
plt.show()

titanic.isnull().sum()

## Cabin and Age have missing values
## Cabin has 687 of 891 missing, better to exclude from the analysis

titanic.info()

## These are some variables we can reject manually as they can not contribute
## to this analysis semantically
titanic_data = titanic.drop(['PassengerId','Name','Ticket','Cabin'], 1)
titanic_data.head()



## Fill in the values based on the class that the passengers travel in
## There seems to be a relation between Age and PClass
## slightly more robust compared to filling in with all the data
## This is of course a choice that one can question and do differently

sns.boxplot(x='Pclass', y='Age', data=titanic_data, palette='hls')
plt.show()

groupAverages = titanic_data.groupby(['Pclass'])['Age'].mean()

def age_approx(cols):
    Age = cols[0]
    Pclass = cols[1]  
    
    if pd.isnull(Age):
        if Pclass == 1:
            return groupAverages[1]
        elif Pclass == 2:
            return groupAverages[2]
        else:
            return groupAverages[3]
    else:
        return Age
    


titanic_data['Age'] = titanic_data[['Age', 'Pclass']].apply(age_approx, axis=1)
titanic_data.isnull().sum()

titanic_data.dropna(inplace=True)
titanic_data.isnull().sum()

## Here, we need to create dummies for the categorical variables, for Sex and Embarked

gender = pd.get_dummies(titanic_data['Sex'],drop_first=True, prefix='sex_')
gender.head()

embark_location = pd.get_dummies(titanic_data['Embarked'],drop_first=True, prefix='Embarked_')
embark_location.head()

## Since we added the dummy variables, we drop the originals
titanic_data.drop(['Sex', 'Embarked'],axis=1,inplace=True)

titanic_dmy = pd.concat([titanic_data,gender,embark_location],axis=1)

titanic_data.head()

## Let's do a check for correlation -- this is a step where you can do the analysis 
## with and without this elimination and observe the effects

colorMap = sns.diverging_palette(145, 280, s=50, l=50, n=200)
sns.heatmap(titanic_dmy.corr(), cmap = colorMap[::-1])  

## Fare and PClass are strongly correlated so we decide to drop one
titanic_dmy.drop(['Pclass'],axis=1,inplace=True)

## This is the step to add an intercept.
## Normally scikit-learn's implementation do not require this and adds this automotically
## We add an intercept constant to be able to use both statsmodels's and scikit-learn's implementations

titanic_dmy['intercept'] = 1.0

titanic_dmy.info()

## Let's also consider normalising Age and Fare so that the effects of scale is mimimized
## You can try with and without this normalisation and observe the impacts on the model

titanic_dmy['Age']=(titanic_dmy['Age']- titanic_dmy['Age'].min())/(titanic_dmy['Age'].max()-titanic_dmy['Age'].min())
#titanic_dmy['Fare']=(titanic_dmy['Fare']- titanic_dmy['Fare'].min())/(titanic_dmy['Fare'].max()-titanic_dmy['Fare'].min())

X = titanic_dmy.ix[:,(1,2,3,4,5,6,7,8)].values

y = titanic_dmy.ix[:,0].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)


LogReg = LogisticRegression(fit_intercept=0)
LogReg.fit(X_train, y_train)

y_pred = LogReg.predict(X_test) 


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)


print(classification_report(y_test, y_pred))

## Let's observe the coefficients -- the best way is to append the names and the coefficients.
## Coefficients tell us about the relation between the variables and the outcome.
## Spot any strong indicators?
coefficients = np.column_stack((np.asarray(titanic_dmy.ix[:,(1,2,3,4,5,6,7,8)].columns), LogReg.coef_.flatten()))
print(coefficients)

## Let's try the same modelling with Statsmodels and see if there are any differences.
## This is mostly an educational step so that you are exposed to both functions, you don't need this normally.
import statsmodels.api as sm 

logit_sm = sm.Logit(y_train, X_train)

# fit the model
result = logit_sm.fit()

# Let's use the very nice summary function of statsmodels
# This is informing us about the model fit. 
# Pseudo R-squ. is a measure to look into and informing us on the overall fit, not really high in this case
print (result.summary())


