# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

loc1 = "C:/Users/hecto/Downloads/AssayData/Assay1.xlsx"
loc2 = "C:/Users/hecto/Downloads/AssayData/Assay2.xlsx"
loc3 = "C:/Users/hecto/Downloads/AssayData/Assay3.xlsx"
loc4 = "C:/Users/hecto/Downloads/AssayData/Assay4.xlsx"
loc5 = "C:/Users/hecto/Downloads/AssayData/Assay5.xlsx"
loc6 = "C:/Users/hecto/Downloads/AssayData/Assay6.xlsx"
loc7 = "C:/Users/hecto/Downloads/AssayData/Assay7.xlsx"

#wb = xlrd.open_workbook(loc)
#sheet =  wb.sheet_by_index(0)

# For row 0 and column 0
#sheet.cell_value(0, 0)
#sheet.nrows

#for row in range(sheet.nrows):
  #  print(sheet.cell_value(row, 2))


#data = pd.read_excel(excel_file)

#Graphs first column of assay. Just one concentration
  
  
#for file in ox.listdir("C:/Users/hecto/Downloads/AssayData/Assay1.xlsx"):
  
#FOR ASSAY1  
print('Assay 1')
x=pd.read_excel(loc1, usecols="B")
g=pd.read_excel(loc1, usecols="C")

plt.plot(x,g)
plt.show()

g2=pd.read_excel(loc1, usecols="C:CT")

plt.plot(x,g2)
plt.show()

y1=pd.read_excel(loc1, usecols="C:Z")
y2=pd.read_excel(loc1, usecols="AA:AX")
y3=pd.read_excel(loc1, usecols="AY:BV")
y4=pd.read_excel(loc1, usecols="BW:CT")

#plt.plot(x,y1,y2,y3,y4)
plt.xlabel('Cycle number')
plt.ylabel('Concentration')
plt.plot(x,y1,color='red')
plt.plot(x,y2,color='green')
plt.plot(x,y3,color='yellow')
plt.plot(x,y4,color='blue')
plt.xlim(0,100)
plt.ylim(2000,4000)
plt.show()

print('------------------------------------------------------------------')

print('Assay 2')

x=pd.read_excel(loc2, usecols="B")
g=pd.read_excel(loc2, usecols="C")

plt.plot(x,g)
plt.show()

g2=pd.read_excel(loc2, usecols="C:CT")

plt.plot(x,g2)
plt.show()

y1=pd.read_excel(loc2, usecols="C:Z")
y2=pd.read_excel(loc2, usecols="AA:AX")
y3=pd.read_excel(loc2, usecols="AY:BV")
y4=pd.read_excel(loc2, usecols="BW:CT")

#plt.plot(x,y1,y2,y3,y4)
plt.xlabel('Cycle number')
plt.ylabel('Concentration')
plt.plot(x,y1,color='red')
plt.plot(x,y2,color='green')
plt.plot(x,y3,color='yellow')
plt.plot(x,y4,color='blue')
plt.show()

print('-----------------------------------------------------------------')

print('Assay 3')
x=pd.read_excel(loc3, usecols="B")
g=pd.read_excel(loc3, usecols="C")

plt.plot(x,g)
plt.show()

g2=pd.read_excel(loc3, usecols="C:CT")

plt.plot(x,g2)
plt.show()

y1=pd.read_excel(loc3, usecols="C:Z")
y2=pd.read_excel(loc3, usecols="AA:AX")
y3=pd.read_excel(loc3, usecols="AY:BV")
y4=pd.read_excel(loc3, usecols="BW:CT")

#plt.plot(x,y1,y2,y3,y4)
plt.xlabel('Cycle number')
plt.ylabel('Concentration')
plt.plot(x,y1,color='red')
plt.plot(x,y2,color='green')
plt.plot(x,y3,color='yellow')
plt.plot(x,y4,color='blue')
plt.show()

print('------------------------------------------------------------------')

print('Assay 4')

x=pd.read_excel(loc4, usecols="B")
g=pd.read_excel(loc4, usecols="C")

plt.plot(x,g)
plt.show()

g2=pd.read_excel(loc4, usecols="C:CT")

plt.plot(x,g2)
plt.show()

y1=pd.read_excel(loc4, usecols="C:Z")
y2=pd.read_excel(loc4, usecols="AA:AX")
y3=pd.read_excel(loc4, usecols="AY:BV")
y4=pd.read_excel(loc4, usecols="BW:CT")

#plt.plot(x,y1,y2,y3,y4)
plt.xlabel('Cycle number')
plt.ylabel('Concentration')
plt.plot(x,y1,color='red')
plt.plot(x,y2,color='green')
plt.plot(x,y3,color='yellow')
plt.plot(x,y4,color='blue')
plt.show()

print('------------------------------------------------------------------')

print('Assay 5')

x=pd.read_excel(loc5, usecols="B")
g=pd.read_excel(loc5, usecols="C")

plt.plot(x,g)
plt.show()

g2=pd.read_excel(loc5, usecols="C:CT")

plt.plot(x,g2)
plt.show()

y1=pd.read_excel(loc5, usecols="C:Z")
y2=pd.read_excel(loc5, usecols="AA:AX")
y3=pd.read_excel(loc5, usecols="AY:BV")
y4=pd.read_excel(loc5, usecols="BW:CT")

#plt.plot(x,y1,y2,y3,y4)
plt.xlabel('Cycle number')
plt.ylabel('Concentration')
plt.plot(x,y1,color='red')
plt.plot(x,y2,color='green')
plt.plot(x,y3,color='yellow')
plt.plot(x,y4,color='blue')
plt.show()

print('------------------------------------------------------------------')

print('Assay 6')

x=pd.read_excel(loc6, usecols="B")
g=pd.read_excel(loc6, usecols="C")

plt.plot(x,g)
plt.show()

g2=pd.read_excel(loc6, usecols="C:CT")

plt.plot(x,g2)
plt.show()

y1=pd.read_excel(loc6, usecols="C:Z")
y2=pd.read_excel(loc6, usecols="AA:AX")
y3=pd.read_excel(loc6, usecols="AY:BV")
y4=pd.read_excel(loc6, usecols="BW:CT")

#plt.plot(x,y1,y2,y3,y4)
plt.xlabel('Cycle number')
plt.ylabel('Concentration')
plt.plot(x,y1,color='red')
plt.plot(x,y2,color='green')
plt.plot(x,y3,color='yellow')
plt.plot(x,y4,color='blue')
plt.show()

print('------------------------------------------------------------------')

print('Assay 7')

x=pd.read_excel(loc7, usecols="B")
g=pd.read_excel(loc7, usecols="C")

plt.plot(x,g)
plt.show()

g2=pd.read_excel(loc7, usecols="C:CT")

plt.plot(x,g2)
plt.show()

y1=pd.read_excel(loc7, usecols="C:Z")
y2=pd.read_excel(loc7, usecols="AA:AX")
y3=pd.read_excel(loc7, usecols="AY:BV")
y4=pd.read_excel(loc7, usecols="BW:CT")

#plt.plot(x,y1,y2,y3,y4)
plt.xlabel('Cycle number')
plt.ylabel('Concentration')
plt.plot(x,y1,color='red')
plt.plot(x,y2,color='green')
plt.plot(x,y3,color='yellow')
plt.plot(x,y4,color='blue')
plt.show()

print('------------------------------------------------------------------')
