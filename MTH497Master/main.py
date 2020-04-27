# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:57:02 2020

@author: hecto
"""

from assay import singleWell, allWells, groupWells, concenWells, tripWells
from regressionfunctions import train, test
<<<<<<< HEAD
from approxPol import approxP
from stddev import stddev
import pandas as pd
import numpy as np

=======
#from approxPol import approxP
from stddev import stddev
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
>>>>>>> df3d405c0a68875ceb8c9d6fea111d0703ca4ce1

loc1 = "Assay1.xlsx"
loc2 = "Assay2.xlsx"
loc3 = "Assay3.xlsx"
loc4 = "Assay4.xlsx"
loc5 = "Assay5.xlsx"
loc6 = "Assay6.xlsx"
loc7 = "Assay7.xlsx"

file = -1 #set file to a random number in order to initialize it

x = pd.read_excel(loc1, usecols="B")
x = np.array(x).reshape(-1)
g2 = pd.read_excel(loc1, usecols="C:CT")

#sigresults holds: [a1, a2, b1, b2, c1, c2, error1, error2]
sigresults = train(x, g2)
test(x, g2, sigresults)

while file != 'stop':
    file=str(input("Please enter an Assay number\n"))
    if(file == '1'):
        file=str(input("Please enter what you would like to see\n"))
        if(file == 'sigmoid'):
            x = pd.read_excel(loc1, usecols="B")
            x = np.array(x).reshape(-1)
            g2 = pd.read_excel(loc1, usecols="C:CT")

            #sigresults holds: [a1, a2, b1, b2, c1, c2, error1, error2]
            sigresults = train(x, g2)
            test(x, g2, sigresults)
        
        elif(file == "standard deviation"):
            file2=str(input("Please enter concentration number\n"))
            if(file2 == '1'):
                stddev(loc1, 1)
            elif(file2 == '2'):
                stddev(loc1, 2)
            elif(file2 == '3'):
                stddev(loc1, 3)
            elif(file2 == '4'):
                stddev(loc1, 4)
            elif(file2 == '5'):
                stddev(loc1, 5)
            elif(file2 == '6'):
                stddev(loc1, 6)
            elif(file2 == '7'):
                stddev(loc1, 7)
            elif(file2 == '8'):
                stddev(loc1, 8)
                
        
        elif(file == "polynomial regression"):
            approxP(loc1)
        elif(file == "single"):
            singleWell(loc1)
        elif(file == "all"):
            allWells(loc1)
        elif(file == "groups"):
            groupWells(loc1)
        elif(file == "concentrations"):
            concenWells(loc1)
        elif(file == "triplicate"):
            tripWells(loc1)
        elif(file == "everything"):
            singleWell(loc1)
            allWells(loc1)
            groupWells(loc1)
            concenWells(loc1)
            tripWells(loc1)
        else:
            print('Invalid. Try again')
            
    elif(file == '2'):
        file=str(input("Please enter which graph you would like to see\n"))
        
        if(file == 'sigmoid'):
            x = pd.read_excel(loc2, usecols="B")
            x = np.array(x).reshape(-1)
            g2 = pd.read_excel(loc2, usecols="C:CT")

            #sigresults holds: [a1, a2, b1, b2, c1, c2, error1, error2]
            sigresults = train(x, g2)
            test(x, g2, sigresults)
        
        
        elif(file == "standard deviation"):
            file2=str(input("Please enter concentration number\n"))
            if(file2 == '1'):
                stddev(loc2, 1)
            elif(file2 == '2'):
                stddev(loc2, 2)
            elif(file2 == '3'):
                stddev(loc2, 3)
            elif(file2 == '4'):
                stddev(loc2, 4)
            elif(file2 == '5'):
                stddev(loc2, 5)
            elif(file2 == '6'):
                stddev(loc2, 6)
            elif(file2 == '7'):
                stddev(loc2, 7)
            elif(file2 == '8'):
                stddev(loc2, 8)
                
        elif(file == "polynomial regression"):
            approxP(loc2)
        elif(file == "single"):
            singleWell(loc2)
        elif(file == "all"):
            allWells(loc2)
        elif(file == "groups"):
            groupWells(loc2)
        elif(file == "concentrations"):
            concenWells(loc2)
        elif(file == "triplicate"):
            tripWells(loc2)
        elif(file == "everything"):
            singleWell(loc2)
            allWells(loc2)
            groupWells(loc2)
            concenWells(loc2)
            tripWells(loc2)
        else:
            print('Invalid. Try again')
            
    elif(file == '3'):
        file=str(input("Please enter which graph you would like to see\n"))
        
        if(file == 'sigmoid'):
            x = pd.read_excel(loc3, usecols="B")
            x = np.array(x).reshape(-1)
            g2 = pd.read_excel(loc3, usecols="C:CT")

            #sigresults holds: [a1, a2, b1, b2, c1, c2, error1, error2]
            sigresults = train(x, g2)
            test(x, g2, sigresults)
        
        elif(file == "standard deviation"):
            file2=str(input("Please enter concentration number\n"))
            if(file2 == '1'):
                stddev(loc3, 1)
            elif(file2 == '2'):
                stddev(loc3, 2)
            elif(file2 == '3'):
                stddev(loc3, 3)
            elif(file2 == '4'):
                stddev(loc3, 4)
            elif(file2 == '5'):
                stddev(loc3, 5)
            elif(file2 == '6'):
                stddev(loc3, 6)
            elif(file2 == '7'):
                stddev(loc3, 7)
            elif(file2 == '8'):
                stddev(loc3, 8)
        
        elif(file == "polynomial regression"):
            approxP(loc3)        
        elif(file == "single"):
            singleWell(loc3)
        elif(file == "all"):
            allWells(loc3)
        elif(file == "groups"):
            groupWells(loc3)
        elif(file == "concentrations"):
            concenWells(loc3)
        elif(file == "triplicate"):
            tripWells(loc3)
        elif(file == "everything"):
            singleWell(loc3)
            allWells(loc3)
            groupWells(loc3)
            concenWells(loc3)
            tripWells(loc3)
        else:
            print('Invalid. Try again')
            
    elif(file == '4'):
        file=str(input("Please enter which graph you would like to see\n"))
        
        if(file == 'sigmoid'):
            x = pd.read_excel(loc4, usecols="B")
            x = np.array(x).reshape(-1)
            g2 = pd.read_excel(loc4, usecols="C:CT")

            #sigresults holds: [a1, a2, b1, b2, c1, c2, error1, error2]
            sigresults = train(x, g2)
            test(x, g2, sigresults)
            
        elif(file == "standard deviation"):
            file2=str(input("Please enter concentration number\n"))
            if(file2 == '1'):
                stddev(loc4, 1)
            elif(file2 == '2'):
                stddev(loc4, 2)
            elif(file2 == '3'):
                stddev(loc4, 3)
            elif(file2 == '4'):
                stddev(loc4, 4)
            elif(file2 == '5'):
                stddev(loc4, 5)
            elif(file2 == '6'):
                stddev(loc4, 6)
            elif(file2 == '7'):
                stddev(loc4, 7)
            elif(file2 == '8'):
                stddev(loc4, 8)
        
        elif(file == "polynomial regression"):
            approxP(loc4)        
        elif(file == "single"):
            singleWell(loc4)
        elif(file == "all"):
            allWells(loc4)
        elif(file == "groups"):
            groupWells(loc4)
        elif(file == "concentrations"):
            concenWells(loc4)
        elif(file == "triplicate"):
            tripWells(loc4)
        elif(file == "everything"):
            singleWell(loc4)
            allWells(loc4)
            groupWells(loc4)
            concenWells(loc4)
            tripWells(loc4)
        else:
            print('Invalid. Try again')
        
    elif(file == '5'):
        file=str(input("Please enter which graph you would like to see\n"))
        
        if(file == 'sigmoid'):
            x = pd.read_excel(loc5, usecols="B")
            x = np.array(x).reshape(-1)
            g2 = pd.read_excel(loc5, usecols="C:CT")

            #sigresults holds: [a1, a2, b1, b2, c1, c2, error1, error2]
            sigresults = train(x, g2)
            test(x, g2, sigresults)
        
        elif(file == "standard deviation"):
            file2=str(input("Please enter concentration number\n"))
            if(file2 == '1'):
                stddev(loc5, 1)
            elif(file2 == '2'):
                stddev(loc5, 2)
            elif(file2 == '3'):
                stddev(loc5, 3)
            elif(file2 == '4'):
                stddev(loc5, 4)
            elif(file2 == '5'):
                stddev(loc5, 5)
            elif(file2 == '6'):
                stddev(loc5, 6)
            elif(file2 == '7'):
                stddev(loc5, 7)
            elif(file2 == '8'):
                stddev(loc5, 8)
        
        elif(file == "polynomial regression"):
            approxP(loc5)        
        elif(file == "single"):
            singleWell(loc5)
        elif(file == "all"):
            allWells(loc5)
        elif(file == "groups"):
            groupWells(loc5)
        elif(file == "concentrations"):
            concenWells(loc5)
        elif(file == "triplicate"):
            tripWells(loc5)
        elif(file == "everything"):
            singleWell(loc5)
            allWells(loc5)
            groupWells(loc5)
            concenWells(loc5)
            tripWells(loc5)
        else:
            print('Invalid. Try again')
        
    elif(file == '6'):
        file=str(input("Please enter which graph you would like to see\n"))
        
        if(file == 'sigmoid'):
            x = pd.read_excel(loc6, usecols="B")
            x = np.array(x).reshape(-1)
            g2 = pd.read_excel(loc6, usecols="C:CT")

            #sigresults holds: [a1, a2, b1, b2, c1, c2, error1, error2]
            sigresults = train(x, g2)
            test(x, g2, sigresults)
        
        elif(file == "standard deviation"):
            file2=str(input("Please enter concentration number\n"))
            if(file2 == '1'):
                stddev(loc6, 1)
            elif(file2 == '2'):
                stddev(loc6, 2)
            elif(file2 == '3'):
                stddev(loc6, 3)
            elif(file2 == '4'):
                stddev(loc6, 4)
            elif(file2 == '5'):
                stddev(loc6, 5)
            elif(file2 == '6'):
                stddev(loc6, 6)
            elif(file2 == '7'):
                stddev(loc6, 7)
            elif(file2 == '8'):
                stddev(loc6, 8)
         
        elif(file == "polynomial regression"):
            approxP(loc6)              
        elif(file == "single"):
            singleWell(loc6)
        elif(file == "all"):
            allWells(loc6)
        elif(file == "groups"):
            groupWells(loc6)
        elif(file == "concentrations"):
            concenWells(loc6)
        elif(file == "triplicate"):
            tripWells(loc6)
        elif(file == "everything"):
            singleWell(loc6)
            allWells(loc6)
            groupWells(loc6)
            concenWells(loc6)
            tripWells(loc6)
        else:
            print('Invalid. Try again')
            
    elif(file == '7'):
        file=str(input("Please enter which graph you would like to see\n"))
        
        if(file == 'sigmoid'):
            x = pd.read_excel(loc7, usecols="B")
            x = np.array(x).reshape(-1)
            g2 = pd.read_excel(loc7, usecols="C:CT")

            #sigresults holds: [a1, a2, b1, b2, c1, c2, error1, error2]
            sigresults = train(x, g2)
            test(x, g2, sigresults)
        
        elif(file == "standard deviation"):
            file2=str(input("Please enter concentration number\n"))
            if(file2 == '1'):
                stddev(loc7, 1)
            elif(file2 == '2'):
                stddev(loc7, 2)
            elif(file2 == '3'):
                stddev(loc7, 3)
            elif(file2 == '4'):
                stddev(loc7, 4)
            elif(file2 == '5'):
                stddev(loc7, 5)
            elif(file2 == '6'):
                stddev(loc7, 6)
            elif(file2 == '7'):
                stddev(loc7, 7)
            elif(file2 == '8'):
                stddev(loc7, 8)
        
        elif(file == "polynomial regression"):
            approxP(loc7)        
        elif(file == "single"):
            singleWell(loc7)
        elif(file == "all"):
            allWells(loc7)
        elif(file == "groups"):
            groupWells(loc7)
        elif(file == "concentrations"):
            concenWells(loc7)
        elif(file == "triplicate"):
            tripWells(loc7)
        elif(file == "everything"):
            singleWell(loc7)
            allWells(loc7)
            groupWells(loc7)
            concenWells(loc7)
            tripWells(loc7)
        else:
            print('Invalid. Try again')
            
    elif(file == 'stop'):
        break

    else:
        print('Invalid. Try again')
