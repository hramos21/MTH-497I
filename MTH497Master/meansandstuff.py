# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:14:56 2020

@author: Sydney's PC
"""

#Created on Thu Feb  6 10:47:53 2020
#@author: hecto

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
from fractions import Fraction as fr

#initialize variables according to assay file
loc1 = "Assay1.xlsx"
loc2 = "Assay2.xlsx"
loc3 = "Assay3.xlsx"
loc4 = "Assay4.xlsx"
loc5 = "Assay5.xlsx"
loc6 = "Assay6.xlsx"
loc7 = "Assay7.xlsx"

file = -1 #set file to a random number in order to initialize it

#graphs single well, all wells, and all wells for each group by color
def assay(location):
    print('Here are the graphs for - '+file)  
    #reads excel file and plots graphs according to columns
    x=pd.read_excel(location, usecols="B")
    g=pd.read_excel(location, usecols="C")
    
    #plots single well
    plt.title('Graph for Single Well - '+file)
    plt.xlabel('Cycle number')
    plt.ylabel('Concentration')
    plt.plot(x,g)
    plt.show()
    
    #reads columns C throught CT from excel file which is all wells in data
    g2=pd.read_excel(location, usecols="C:CT")
    
    plt.title('Graph for all wells - '+file)
    plt.xlabel('Cycle number')
    plt.ylabel('Concentration')
    plt.plot(x,g2)
    plt.show()
    
    #seperates y values into the 4 different groups
    y1=pd.read_excel(location, usecols="C:Z")
    y2=pd.read_excel(location, usecols="AA:AX")
    y3=pd.read_excel(location, usecols="AY:BV")
    y4=pd.read_excel(location, usecols="BW:CT")
    
    plt.xlabel('Cycle number')
    plt.ylabel('Concentration')
    plt.title('Graph for 4 groups - '+file)
    plt.plot(x,y1,color='red')#labels groups according to color
    plt.plot(x,y2,color='green')
    plt.plot(x,y3,color='yellow')
    plt.plot(x,y4,color='blue')
    #plt.xlim(0,100)
    #plt.ylim(2000,4000)
    plt.show()
    
    nar = g2.to_numpy() #convert g2 to numpy array
    
    #make an array of colors to play with
    colors = ["red", "yellow", "blue", "green", "orange", "purple", "pink", "teal", "goldenrod"]
    
    plt.title('Graph for 8 concentrations - '+file)
    plt.xlabel('Cycle number')
    plt.ylabel('Concentration')
    for k in range(8):
        #indices of each column for a given concentration
        indices = [k*3+0, k*3+1, k*3+2, k*3+24, k*3+25, k*3+26, k*3+48, 
                   k*3+49, k*3+50, k*3+72, k*3+73, k*3+74]
        plt.plot(x, nar[:, indices], color = colors[k] ) #plot each concentration 
    plt.show()
    
    #make an empty array of wells
    wells = np.zeros((nar.shape[0], int(nar.shape[1]/3)))
    
    #to graph averaged triplicates as one well
    #start at 0, go to the last column, skip by 3s
    for j in range(0, nar.shape[1], 3):
        triplicate = nar[:, j: j + 3] #get all the rows in a slice of 3 cols
        means = np.mean(triplicate, axis=1) #find the mean of all the rows
        meansupright = np.vstack(means) # shift the horizontal array into vert
        #take our means and stick it in our empty array
        wells[:, int(j/3): int(j/3) + 1] = meansupright 
        
    plt.title('Graph for Triplicates - '+file)
    plt.xlabel('Cycle number')
    plt.ylabel('Concentration')
    plt.plot(x, wells)
    plt.show()
    
    print('---------------------------------------------------------------')
    return

def stddev(location, concentration):
    x=pd.read_excel(location, usecols="B")
    
    g2=pd.read_excel(location, usecols="C:CT")
    nar = g2.to_numpy() #convert g2 to numpy array
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
    #print(deriv)
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
    plt.plot(x, secondderiv, color = 'blue' ) #plot the derivative 
    plt.show()
    plt.plot(x2[:minlocation+10], nar[:minlocation+10,indices], color = 'red')
    plt.axvline(x = minlocation)
    plt.show()

    return

stddev(loc5, 6)
#stddev(loc2, 2)
#stddev(loc2, 3)

#will always ask for input unless user types 'stop'
while file != 'stop':
    file=str(input("Please enter the Assay you would like to see\n"))

    if(file == 'assay1'):
        assay(loc1)
    
    elif(file == 'assay2'):
        assay(loc2)
        
    elif(file == 'assay3'):
        assay(loc3)
    
    elif(file == 'assay4'):
        assay(loc4)
    
    elif(file == 'assay5'):
        assay(loc5)
    
    elif(file == 'assay6'):
        assay(loc6)
    
    elif(file == 'assay7'):
        assay(loc7)
    elif(file == 'all'):
        print('ASSAY 1')
        assay(loc1)
        print('ASSAY 2')
        assay(loc2)
        print('ASSAY 3')
        assay(loc3)
        print('ASSAY 4')
        assay(loc4)
        print('ASSAY 5')
        assay(loc5)
        print('ASSAY 6')
        assay(loc6)
        print('ASSAY 7')
        assay(loc7)
    else:
        print('Invalid assay. Try again')