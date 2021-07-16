# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:31:02 2019

@author: Prinzessin
"""

# mpg = miles per gallon

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import statsmodels.api as sm

def import_data():
    df = pd.read_csv("auto.csv")
    filter = df.loc[df["horsepower"] != '?']
    df['horsepower'] = filter.horsepower.astype(int)
    return df

def import_data_v2():
    df = pd.read_csv("auto.csv", sep=',', na_values='?')
    # as we don't even import ? horsepower is not saved as a string
    df = df.dropna() # delete na values
    return df  
    
def plot_linear_reg(data, col):
    name = str(col)
    # nsample = 100
    X = data[[name]] # we need a list
    y = data['mpg']
    result = sm.OLS(y, sm.add_constant(X)).fit()
    
    model = result.params[0] + result.params[1] * data[name] 
    # need to use fit here again? - train data
    print("model:", name)
    print(str(model))
    
    # dim
    plt.scatter(data[name], data['mpg'])
    plt.plot(data[name], model)
    
    # chrisy - dodgy version
    ax = data.plot(x=name, y='mpg', kind='scatter')
    ax.plot(data[name], model)

def print_relationships(var, mpg, column):
    plt.plot(mpg, var, 'bo', label='Variable')
    plt.title('Relationship: mpg and ' + str(column))
    plt.xlabel('mpg')
    plt.ylabel(column)
    plt.legend()
    # plt.show()
    # save graph
    plt.savefig(str(column))
    plt.clf() # clear plot
    
""" Ordinary Least Squares OLS 
R-Squared should become 1
P should be 0 - everything over 0.05 should be removed from X data
"""
def r_squared(data):
    # nsample = 100
    X = data[['model_year', 'origin', 'weight']]
    y = data['mpg']
    result = sm.OLS(y, sm.add_constant(X)).fit()
    print(result.summary())
    
    model = result.params[0] + result.params[1] * data['model_year'] + result.params[2] * data['origin'] + result.params[3] * data['weight']
    # need to use the fit here again ? - train data
    print("model: year, origin, weight:")
    print(model)      

""" ---------- function calls ---------- """
df = import_data_v2() # df = import_data()
#print(df)

for column in df:
    if column != 'name' and column != 'mpg':
        print_relationships(df['mpg'], df[column], column)
        
plot_linear_reg(df, 'horsepower')

r_squared(df)

