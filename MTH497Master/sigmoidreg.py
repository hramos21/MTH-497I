# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:38:49 2020
this is using Claire's edited version of logisttic regression
@author: Sydney's PC
"""

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from scipy.optimize import curve_fit
loc1 = "Assay1.xlsx"
loc2 = "Assay2.xlsx"
loc3 = "Assay3.xlsx"
loc4 = "Assay4.xlsx"
loc5 = "Assay5.xlsx"
loc6 = "Assay6.xlsx"
loc7 = "Assay7.xlsx"


def sigmoid(x, a, b):
     y = 1 / (1 + np.exp(-b*(x-a)))
     return y

def approxLog(location):
    x=pd.read_excel(location, usecols="B")
    x1 = np.array(x).reshape(-1)
    g=pd.read_excel(location, usecols="C")
    g2 = np.array(g).reshape(-1) 
    g2 = [item - np.min(g2) for item in g2] #downshift to 0
    g2 = [item/(np.max(g2)-np.min(g2)) for item in g2] #normalize 
    g2 = [item if item > 0 else 0.0000001 for item in g2] #if item is exactly zero, make it not that
    parameters, pcov = curve_fit(sigmoid, x1, g2, p0=[110, 0.001])
    print(parameters)
    print(pcov)
    y = sigmoid(x1, *parameters)
    #print(y)
    #print(g2)
    error = [0]*241
    for i in range (0,241):
        error[i] = (g2[i] - y[i])/g2[i]
     
    print(error)
    #logmod = LogisticRegression().fit(x1, g2)
    #print(logmod.intercept_)
    #print(logmod.coef_)
    #prob = logmod.predict_proba(x1[3:])
    #pred = logmod.predict(x1)
    #print(pred)
    
    
    plt.plot(x1,g2)##actual graph
    plt.plot(x1, y)##regression line 
    plt.savefig('pred.png')
    plt.show()
    plt.clf()
    return

approxLog(loc1)

##to do: run prediction of what each rfu would be and compare with actual then r^2
##error bars, total relative error
##minimize error
##may need to add c parameter to sigmoid function to bump line right or left
