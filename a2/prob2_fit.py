import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
CSCI 635: Introduction to Machine Learning
Problem 2: Polynomial Regression &

@author/lecturer - Alexander G. Ororbia II

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

# NOTE: you will need to tinker with the meta-parameters below yourself
#       (do not think of them as defaults by any means)
# meta-parameters for program
trial_name = 'prob2' # will add a unique sub-string to output of this program
degree = 7 # p, order of model
beta = 0.0001 # regularization coefficient
alpha = 0.1 # step size coefficient
eps = 0.00001 # controls convergence criterion
n_epoch = 10000 # number of epochs (full passes through the dataset)

# begin simulation

def regress(X, theta):
    ############################################################################
	# WRITEME: write your code here to complete the routine
	b, w = theta
	f = np.dot(X,w.T) + b
	return f
    ############################################################################

def gaussian_log_likelihood(mu, y):
    ############################################################################
	# WRITEME: write your code here to complete the routine
	return np.sum((mu-y)**2)
    ############################################################################

def computeCost(X, y, theta, beta): ## loss is now Bernoulli cross-entropy/log likelihood
    ############################################################################
	# WRITEME: write your code here to complete the routine
	size = X.shape[0]
	b, w = theta
	f = np.dot(X,w.T) + b
	term = gaussian_log_likelihood(f, y)
	optimizer= term / (2 * size)
	reg= (beta*np.sum(w)**2)/(2 * size)
	return optimizer+reg
    ############################################################################

def computeGrad(X, y, theta, beta):
    ############################################################################
	# WRITEME: write your code here to complete the routine (
	# NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
	dL_dfy = None # derivative w.r.t. to model output units (fy)
	b, w = theta
	size = X.shape[0]
	f = np.dot(X,w.T) + b
	dL_db = np.sum(f-y)/size # derivative w.r.t. model weights w
	dL_dw = (np.dot((f-y).T,X)/size) +((beta*w)/size) # derivative w.r.t model bias b
	nabla = (dL_db, dL_dw) # nabla represents the full gradient
	return nabla
    ############################################################################

path = os.getcwd() + '/data/prob2.dat'
data = pd.read_csv(path, header=None, names=['X', 'Y'])

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert from data frames to numpy matrices
X = np.array(X.values)
y = np.array(y.values)

############################################################################
# apply feature map to input features x1
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you
#          could use a loop and array concatenation)
############################################################################

dss=X[:,-1]
for i in range(2,degree+1):
	xnew=X[:, 0].T ** i
	xnew=xnew.reshape((X.shape[0], 1))
	X=np.concatenate((X,xnew),axis=1)
# convert to numpy arrays and initalize the parameter array theta
w = np.zeros((1,X.shape[1]))
b = np.array([0])
theta = (b, w)
print("X ",X.shape)
print("w ",w.shape)
L = computeCost(X, y, theta, beta)
halt = np.inf # halting variable (you can use these to terminate the loop if you have converged)
print("-1 L = {0}".format(L))
i = 0
cost = []
cost.append(L)
while(i < n_epoch and halt >=eps):
	dL_db, dL_dw = computeGrad(X, y, theta, beta)
	b = theta[0]
	w = theta[1]
    ############################################################################
	# update rules go here...
	# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
    ############################################################################
	b=b-(alpha*dL_db)
	w=w-(alpha*dL_dw)
	theta = (b, w)
	L = computeCost(X, y, theta, beta)
	cost.append(L)
    ############################################################################
	# WRITEME: write code to perform a check for convergence (or simply to halt early)
    ############################################################################
	if len(cost)>=2:
		halt = cost[-2]-cost[-1]
		# print(halt)
	print(" {0} L = {1}".format(i,L))
	i += 1
# print parameter values found after the search
print("w = ",w)
print("b = ",b)
# print("X = ",X.shape)
# print("w = ",w.shape)
# print("b = ",b.shape)
halt = cost[-2]-cost[-1]
if  halt <=eps:
	print("Model initial epochs set at ",n_epoch)
	print("Convergence happened at epoch ",i-1)


kludge = 0.25
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_feat = np.expand_dims(X_test, axis=1) # we need this otherwise, the dimension is missing (turns shape(value,) to shape(value,value))
for i in range(2,degree+1):
	xnew=X_feat[:, 0].T ** i
	xnew=xnew.reshape((X_feat.shape[0], 1))
	X_feat=np.concatenate((X_feat,xnew),axis=1)
############################################################################
# apply feature map to input features x1
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you
#          could use a loop and array concatenation)
############################################################################
outputpath = os.getcwd() + '/out/'
plt.plot(X_test, regress(X_feat, theta), label="Model")
plt.scatter(X[:,0], y, edgecolor='g', s=20, label="Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
plt.legend(loc="best")

############################################################################
# WRITEME: write your code here to save plot to disk (look up documentation or
#          the inter-webs for matplotlib)
############################################################################
plt.savefig(outputpath+trial_name+"_output_scatter_with_regression"+'.jpeg')
plt.show()

### Loss plot
plt.plot([j for j in range(len(cost))], cost, label="training loss curve")
plt.xlabel("EPOCHS")
plt.ylabel("LOSS")
plt.savefig(outputpath+trial_name+"_training_loss_curve"+'.jpeg')
plt.show()

