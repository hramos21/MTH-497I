# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:19:28 2020

@author: hecto
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

loc1 = "Assay1.xlsx"
loc2 = "Assay2.xlsx"
loc3 = "Assay3.xlsx"
loc4 = "Assay4.xlsx"
loc5 = "Assay5.xlsx"
loc6 = "Assay6.xlsx"
loc7 = "Assay7.xlsx"

def stddev(location, concentration):
    x=pd.read_excel(location, usecols="B")
    
    g2=pd.read_excel(location, usecols="C:CT")
    nar = g2.to_numpy() #convert g2 to numpy array
    con = concentration
    
    #indices of each column for a given concentration
    indices = [(con-1)*3+0, (con-1)*3+1, (con-1)*3+2, (con-1)*3+24, (con-1)*3+25, (con-1)*3+26, (con-1)*3+48, 
                   (con-1)*3+49, (con-1)*3+50, (con-1)*3+72, (con-1)*3+73, (con-1)*3+74]
    
    #given one specified assay and concentration, plot the mean of that concentration 
    means1 = np.mean(nar[:, indices], axis = 1)
    standev = np.std(nar[:, indices], axis = 1)
    means2 = np.array(means1)
    deriv = np.gradient(means2)
    deriv2 = np.gradient(deriv)
    print(deriv)
    x1=np.linspace(0,240,1)
    #print('Mean =',means1)
    plt.title('Graph for Means')
    plt.xlabel('Cycle number')
    plt.ylabel('Concentration')
    plt.plot(x, means1, color = 'red' ) #plot each concentration 
    plt.show()
    plt.title('Graph for Standard Deviation')
    plt.xlabel('Cycle number')
    plt.ylabel('Concentration')
    plt.plot(x, standev, color = 'green' )
    plt.show()
    plt.title('Graph for Derivatives')
    plt.xlabel('Cycle number')
    plt.ylabel('Concentration')
    plt.plot(x, deriv, color = 'blue' ) #plot the derivative 
    plt.show()
    plt.title('Second derivative graph')
    plt.xlabel('Cycle number')
    plt.ylabel('Concentration')
    plt.plot(x, deriv2, color = 'green')
    plt.show()
    return

stddev(loc1, 4)

# def stddev(location, concentration):
#     x=pd.read_excel(location, usecols="B")
    
#     g2=pd.read_excel(location, usecols="C:CT")
#     nar = g2.to_numpy() #convert g2 to numpy array
#     con = concentration
    
#     #indices of each column for a given concentration
#     indices = [(con-1)*3+0, (con-1)*3+1, (con-1)*3+2, (con-1)*3+24, (con-1)*3+25, (con-1)*3+26, (con-1)*3+48, 
#                    (con-1)*3+49, (con-1)*3+50, (con-1)*3+72, (con-1)*3+73, (con-1)*3+74]
    
#     #given one specified assay and concentration, plot the mean of that concentration 
#     means1 = np.mean(nar[:, indices], axis = 1)
#     standev = np.std(nar[:, indices], axis = 1)
#     #print('Mean =',means1)
#     plt.plot(x, means1, color = 'red' ) #plot each concentration 
#     plt.show()
#     plt.plot(x, standev, color = 'green' )
#     plt.show()
#     return