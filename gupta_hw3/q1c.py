"""
FileName: q1c.py
Author: prakhar gupta pg9349
Description: Create a multinomial linear model for iris data
"""


import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from helper_functions import *
np.random.seed(1)
def mini_batch_generator(X,y,batchsize=32):
	"""
	Create a mini batchs for the data
	:param X: data
	:param y: label
	:param batchsize: batch size
	:return:
	"""
	xx=X.shape[1]
	yy=y.shape[1]

	all_data = np.hstack((X, y))
	# print(all_data.shape)
	np.random.shuffle(all_data)
	X=all_data[:,0:xx]

	y = all_data[:,xx:]

	batches=[]
	batchcount=X.shape[0]//batchsize
	for i in range(batchcount):
		x1=X[batchsize*i:batchsize*i+batchsize,:]
		y1=y[batchsize*i:batchsize*i+batchsize]
		batches.append((x1,y1))
	x1 = X[batchsize * (i+1):, :]
	y1 = y[batchsize * (i+1):]
	batches.append((x1, y1))
	return batches

beta = 1e-3 # regularization coefficient
alpha = 0.02 # step size coefficient
n_epoch = 2000 # number of epochs (full passes through the dataset)
eps = 0# controls convergence criterion

# begin simulation

path = os.getcwd() + '/iris_train.dat'
data2 = pd.read_csv(path, header=None)
X=data2.to_numpy()

path = os.getcwd() + '/iris_test.dat'
dataval = pd.read_csv(path, header=None)
Xval=dataval.to_numpy()

cols = data2.shape[1]
X2 = data2.iloc[:,0:-1]
y2 = data2.iloc[:,-1]
y3=y2.values
y3=np.vstack(y3)
y2=pd.get_dummies(y2)
print(X2.shape)

X2val = dataval.iloc[:,0:-1]
y2val = dataval.iloc[:,-1]
y2val=pd.get_dummies(y2val)


# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)
y2 = np.array(y2.values)

X2val = np.array(X2val.values)
y2val = np.array(y2val.values)

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
valcost=[]
cost.append(L)
while(i < n_epoch):
	batches=mini_batch_generator(X2, y2, batchsize=32)
	counter=1
	for minix,miniy in batches:
		# print(minix.shape)
		# print(miniy.shape)
		dL_db, dL_dw = computeGrad(minix, miniy, theta2, beta)
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
		Lval=computeCost(X2val,y2val,theta2,beta)
		# print(w)
		############################################################################
		# WRITEME: write code to perform a check for convergence (or simply to halt early)
		############################################################################
		cost.append(L)
		valcost.append((Lval))
		if len(cost)>=2:
			halt = cost[-2]-cost[-1]
			# print(halt)
		print(" 	Batch No {0} Loss = {1}".format(counter, L))
		counter+=1
	print("After Complete epoch {0} Loss = {1}".format(i,L))
	i += 1
# print parameter values found after the search

print("w = ",w)
print("b = ",b)
halt = cost[-2]-cost[-1]
# if  halt <=eps:
# 	print("Model initial epochs set at ",n_epoch)
# 	print("Convergence happened at epoch ",i-1)
############################################################################
predictions = predict(X2, theta2)
predictionsval = predict(X2val, theta2)
# print(predictions)
# print(predictions)
# WRITEME: write your code here calculate your actual classification error (using the "predictions" variable)
y3=np.argmax(y2,axis=1)
N = predictions.shape[0]
y=predictions

y3val=np.argmax(y2val,axis=1)
Nval = predictionsval.shape[0]
yval=predictionsval
print("________________________________________________")
print("Train Accuracy")
print("________________________________________________")
# print(predictions)
accuracy = (y3 == y).sum() / N
print('Accuracy = {0}%'.format(accuracy * 100.))
err = 1-accuracy
print ('Error = {0}%'.format(err * 100.))

print("________________________________________________")
# print(predictions)
print("Validation Accuracy")
print("________________________________________________")
accuracyval = (y3val == yval).sum() / Nval
print('Accuracy = {0}%'.format(accuracyval * 100.))
err = 1-accuracyval
print ('Error = {0}%'.format(err * 100.))




# x=mini_batch_generator(X2,y2)
plt.plot(cost, label='Training Loss')
plt.plot(valcost,label="Validation Loss")
plt.title("LOSS CURVE")
plt.xlabel("Iterations i.e (Epochs* No of Batches)")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig("Losscurveq1c.jpeg")
plt.show()
