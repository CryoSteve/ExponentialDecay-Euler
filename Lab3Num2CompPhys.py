import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot as plt 
tArray = np.empty([1])
Narray = np.empty([1])

#fig, ax = plt.subplots()
#fig2, bx = plt.subplots()
def analyticExp(n0,t):
  return(n0*math.exp(-t))

h = float(0.1)
numofSteps = float((10-0)/h)
def function(N,t,h): 
  return float(-1.0*N)
t0 = float(input("Enter your initial value of t"))
N0 = 1000.000 
tArray = np.append(tArray,t0)
Narray = np.append(Narray,N0)
tArray = np.delete(tArray,0)
Narray = np.delete(Narray, 0)
#print('Your value of steps is'+ str(numofSteps))
for i in range(int(numofSteps)):
  m = function(N0,t0,h)
  N1 = N0 + h*m
  t1 = t0+h
  print("T"+str(i)+" = "+str(t1)+", N"+str(i)+" = "+str(N1))
  N0 = N1
  t0 = t1
  tArray = np.append(tArray,t0)
  Narray = np.append(Narray,N0)
#print(tArray)
#print(Narray)
N2array = np.empty([0])
for i in range(len(tArray)):
  N2array = np.append(N2array,analyticExp(1000.0,tArray[i]))
ErrorArray = np.empty([0])
for i in range(len(tArray)):
  ErrorArray = np.append(ErrorArray,N2array[i]-Narray[i])
#print (N2array)
df = pd.DataFrame()
df['t'] = tArray.tolist()
df['N(t)'] = Narray.tolist()
df['N*(t)'] = N2array.tolist()
df['E(t)'] = ErrorArray.tolist()
print(df.to_string())
h2=0.01

#ax.scatter(tArray,Narray)
#bx.scatter(tArray,N2array)
#bx.set_title("Analytic Results")
#ax.set_title("Numerical Results")
plt.plot(tArray,Narray,color="g",label = 'numerical(h = 0.1)')
plt.plot(tArray,N2array,color="b",label ='analytical')
plt.xlabel("t")
plt.ylabel("N(t)")
plt.title("Exponential Decay")
plt.legend()
plt.show()


#ax = df.plot(x='t',y='x(t)',figsize=(10,16),color='blue',label='x(t)',linewidth=3)