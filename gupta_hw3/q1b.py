"""
FileName: q1b.py
Author: prakhar gupta pg9349
Description: Create a multinomial linear model for spiral data
"""


import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from helper_functions import *



beta = 1e-3 # regularization coefficient
alpha = 0.01 # step size coefficient
n_epoch = 10000 # number of epochs (full passes through the dataset)
eps = 0# controls convergence criterion

# begin simulation

path = os.getcwd() + '/spiral_train.dat'
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


h = 0.001
cmap='RdBu'

x_min, x_max = X2[:,0].min() - 100*h, X2[:,0].max() + 100*h
y_min, y_max = X2[:,1].min() - 100*h, X2[:,1].max() + 100*h
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# print(np.c_[xx.ravel(), yy.ravel()])
Z = predict(np.c_[xx.ravel(), yy.ravel()],theta2)
Z = Z.reshape(xx.shape)

plt.plot(cost, label='Training Loss')
plt.title("LOSS CURVE")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig("Losscurveq2b.jpeg")
plt.show()



plt.figure(figsize=(7,7))
plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.9)
plt.contour(xx, yy, Z, colors='k', linewidths=1)
plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap, edgecolors='k');
plt.title("Spiral data with decision boundary")
plt.savefig("decision_boundaryq1b.png")
plt.show()
