"""
FileName: q1a.py
Author: prakhar gupta pg9349
Description: Create a multinomial linear model for XOR
"""


import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from helper_functions import *



beta = 1e-4 # regularization coefficient
alpha = 0.01 # step size coefficient
n_epoch = 10000 # number of epochs (full passes through the dataset)
eps =  0.00001# controls convergence criterion

# begin simulation

path = os.getcwd() + '/xor.dat'
data2 = pd.read_csv(path, header=None)
X=data2.to_numpy()

cols = data2.shape[1]
X2 = data2.iloc[:,0:-1]
y2 = data2.iloc[:,-1]
y3=y2.values
y3=np.vstack(y3)
y2=pd.get_dummies(y2)
np.random.seed(1)
# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)
y2 = np.array(y2.values)
w = np.random.rand(X2.shape[1],y2.shape[1])
# # Uncomment for zero intialization
# w = np.zeros((X2.shape[1],y2.shape[1]))
w1=w
b = np.zeros((1,y2.shape[1]))

print(w)
print(b)

theta2 = (b, w)

L = computeCost(X2, y2, theta2, beta)

halt = np.inf # halting variable (you can use these to terminate the loop if you have converged)
print("-1 L = {0}".format(L))

i = 0
cost=[]
cost.append(L)
while(i < n_epoch and halt >=eps):
	dL_db, dL_dw = computeGrad(X2, y2, theta2, beta)
	b = theta2[0]
	w = theta2[1]
    ############################################################################
	# update rules go here...
	# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
    ############################################################################
	b=b-(alpha*dL_db)
	w=w-(alpha*dL_dw)
	theta2 = (b, w)
	L = computeCost(X2, y2, theta2, beta)
	# print(w)
    ############################################################################
	# WRITEME: write code to perform a check for convergence (or simply to halt early)
    ############################################################################
	cost.append(L)
	if len(cost)>=2:
		halt = cost[-2]-cost[-1]
		# print(halt)

	print(" {0} L = {1}".format(i,L))
	i += 1
# print parameter values found after the search

print("w = ",w)
print("b = ",b)
halt = cost[-2]-cost[-1]
if  halt <=eps:
	print("Model initial epochs set at ",n_epoch)
	print("Convergence happened at epoch ",i-1)
############################################################################
predictions = predict(X2, theta2)
# print(predictions)
# print(predictions)
# WRITEME: write your code here calculate your actual classification error (using the "predictions" variable)
y3=np.argmax(y2,axis=1)
N = predictions.shape[0]
y=predictions

print("-------")
# print(predictions)
accuracy = (y3 == y).sum() / N
print('Accuracy = {0}%'.format(accuracy * 100.))
err = 1-accuracy
print ('Error = {0}%'.format(err * 100.))


plt.plot(cost, label='Training Loss')
plt.title("LOSS CURVE")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig("Losscurveq2a.jpeg")
plt.show()



print("_____________________________________________________")
print("Gradient Checking Start")



"""
Gradient checking
Coding derivatives using limits proof

"""

delta=1e-4
w = np.zeros((X2.shape[1],y2.shape[1]))
# w=w1
b = np.zeros((1,y2.shape[1]))

secant_derivative_bias=np.zeros(b.shape)
secant_derivative_weights=np.zeros(w.shape)
originalw=w
originalb=b
theta = (b, w)
dL_db, dL_dw = computeGrad(X2, y2, theta, beta)
# theta = (b, w)
print("_____________________________________________________")
print("Checking for derivatives i.e bias")
for x, y in np.ndindex(b.shape):
	b = originalb
	b[x,y]=b[x,y]-delta
	theta = (b, w)
	term1=computeCost(X2, y2, theta, beta)
	b=originalb
	b[x,y]=b[x,y]+delta
	theta = (b, w)
	term2=computeCost(X2, y2, theta, beta)
	final=(term2-term1)/(delta*2)
	secant_derivative_bias[x,y]=final
	# np.insert(secant_derivative_bias,(x,y), final)
# print(secant_derivative_bias)
# print(dL_db)
print("_____________________________________________________")
print("Individual biases")
extra=abs(secant_derivative_bias-dL_db)<=1e-4
extra=extra.tolist()
for i in extra:
	for j in i:
		if j==True:
			print("CORRECT",end=" ")
		else:
			print(j,end=" ")
	print()
# print(abs(secant_derivative_bias-dL_db)<=1e-4)
if (abs(secant_derivative_bias-dL_db)<=1e-4).all():
	print("Bias Derivative Correct")

print("_____________________________________________________")
print("Checking for derivatives i.e weights")

for x, y in np.ndindex(w.shape):
	w = originalw
	w[x,y]=w[x,y]-delta
	theta = (b, w)
	term1=computeCost(X2, y2, theta, beta)
	w=originalw
	w[x,y]=w[x,y]+delta
	theta = (b, w)
	term2=computeCost(X2, y2, theta, beta)
	final=(term2-term1)/(delta*2)
	secant_derivative_weights[x,y]=final
	# np.insert(secant_derivative_bias,(x,y), final)
# print(secant_derivative_weights)
# print(dL_dw)
print("_____________________________________________________")
print("Individual Weights")

extra=abs(secant_derivative_weights-dL_dw)<=1e-4
extra=extra.tolist()
for i in extra:
	for j in i:
		if j==True:
			print("CORRECT",end=" ")
		else:
			print(j,end=" ")
	print()

if (abs(secant_derivative_weights-dL_dw)<=1e-4).all():
	print("Weights Derivative Correct")

