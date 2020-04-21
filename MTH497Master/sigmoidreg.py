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


def sigmoid(x, a, b, c):
     y = 1 / (1 + c*np.exp(-b*(x-a))) 
     return y

def predictedsig(x, a):
    b = 5.16E-2
    c = 1.23E2
    y1 = sigmoid(x, a, b, c)
    return y1
def approxLog(location, concentration):
    x=pd.read_excel(location, usecols="B")
    x1 = np.array(x).reshape(-1)
    g3=pd.read_excel(location, usecols="C:CT")
    nar = g3.to_numpy() #convert g2 to numpy array
    con = concentration
    x2 = x.to_numpy()
    
    #indices of each column for a given concentration
    indices = [(con-1)*3+0, (con-1)*3+1, (con-1)*3+2, (con-1)*3+24, (con-1)*3+25, (con-1)*3+26, (con-1)*3+48, 
                   (con-1)*3+49, (con-1)*3+50, (con-1)*3+72, (con-1)*3+73, (con-1)*3+74]
    
    #given one specified assay and concentration, plot the mean of that concentration 
    means1 = np.mean(nar[:, indices], axis = 1)
    standev = np.std(nar[:, indices], axis = 1)
    means2 = np.array(means1)
    deriv = np.gradient(means2)
    secondderiv = np.gradient(deriv)
    minlocation = int(np.where(secondderiv == np.min(secondderiv))[0])
    g=pd.read_excel(location, usecols="C")
    g2 = np.array(means1).reshape(-1) 
    g2 = [item - (np.min(g2) - 0.0000001) for item in g2] #downshift to 0
    g2 = [item/(np.max(g2)-np.min(g2)) for item in g2] #normalize 
    #g2 = [item if item > 0 else 0.0000001 for item in g2] #if item is exactly zero, make it not that
    parameters, pcov = curve_fit(sigmoid, x1, g2, p0=[110, 0.001, 100])
    print(parameters)
    print(pcov)
    y = sigmoid(x1, *parameters)
    #y1 = predictedsig(x1, 15)
    #print(y)
    #print(g2)
    error = 0
    errorpre = 0
    for row in range (0,len(g2)):
        error += (g2[row] - y[row])**2
    error = np.sqrt(error/len(g2))
    ##note: change range to vector -50:.5:50
    for step in range (-50, 50):
        errorpre = 0
        y1 = predictedsig(x1, step)
        #plt.plot(x1,y1)
        for row in range (0,len(g2)):         
            errorpre += (g2[row] - y1[row])**2
        errorpre = np.sqrt(errorpre/len(g2))
        if errorpre < error + .01:
            print(errorpre)
            break
    
     
    print(error) #standard error
    yerr = .1 + .2*np.sqrt(x)
    xerr = .1 + yerr
    plt.figure()
    plt.errorbar(x, y, xerr, yerr)
    plt.title("simple errorbars")
    #print(errorpre) #standard error
    #logmod = LogisticRegression().fit(x1, g2)
    #print(logmod.intercept_)
    #print(logmod.coef_)
    #prob = logmod.predict_proba(x1[3:])
    #pred = logmod.predict(x1)
    #print(pred)
    
    
    plt.plot(x1,g2)##actual graph
    plt.plot(x1, y)##regression line 
    plt.plot(x1,y1)
    plt.savefig('pred.png')
    plt.show()
    plt.clf()
    return

approxLog(loc1, 5)

##to do: run prediction of what each rfu would be and compare with actual then r^2
##error bars, total relative error
##minimize error
##may need to add c parameter to sigmoid function to bump line right or left
