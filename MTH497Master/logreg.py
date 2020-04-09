# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:45:11 2020
This is our logistic regression function
@author: Sydney Wells
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
loc1 = "Assay1.xlsx"
loc2 = "Assay2.xlsx"
loc3 = "Assay3.xlsx"
loc4 = "Assay4.xlsx"
loc5 = "Assay5.xlsx"
loc6 = "Assay6.xlsx"
loc7 = "Assay7.xlsx"

def approxLog(location):
    x=pd.read_excel(location, usecols= "B")
    #x1 = x.iloc[0:241, :]
    x1 = np.array(x).reshape(-1,1)
    g=pd.read_excel(location, usecols="C")
    myline = np.linspace(0,241,241).reshape(-1,1)
    ##take our boring pandas array and make it a numpy array, then kick out the extra dimension
    g2 = np.array(g,dtype="int").reshape(-1)
    print(x1.shape)
    print(g2.shape)
    logmod = LogisticRegression(solver='liblinear', random_state=0, fit_intercept = False).fit(myline, g2)
    #logline.fit
    logmod.classes_
    #interc = logmod.intercept_.shape
    print(logmod.intercept_)
    print(logmod.coef_.shape)
    prob = logmod.predict_proba(x1)
    print(prob)
    print(logmod.predict(x1).shape)
    #print(logmod.predict(x1))
    #print(prob)
    #logline = LogisticRegression().fit(x1, g2)##we have a regression line! 
    ##its not very good tho
    
    
    #plt.plot(x,g)##actual graph
    #plt.plot(x1, interc)##regression line 
    plt.plot(prob[1,:])
    plt.show()
    return
approxLog(loc1)