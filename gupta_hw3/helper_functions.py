"""
FileName: helperfunctions.py
Author: prakhar gupta pg9349
Description: functions to train models for part 1a,1b,1c
"""
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd




def softmax(z):
    ############################################################################
	# WRITEME: write your code here to complete the routine
	# print(np.sum(np.exp(z),axis=1))
	# print(np.exp(z))
	denom=np.sum(np.exp(z), axis=1)
	# print(denom.shape)
	denom=np.reshape(denom,(denom.shape[0],1))
	# print(denom)
	return np.exp(z) / denom
    ############################################################################



def predict(X, theta):
    ############################################################################
	# WRITEME: write your code here to complete the routine
	p=regress(X,theta)
	# return p
	# print(p)
	return np.argmax(p,axis=1)
    # ############################################################################


def regress(X, theta):
    ############################################################################
	# WRITEME: write your code here to complete the routine
	b, w = theta
	f = np.dot(X,w) + b
	return softmax(f)


def log_likelihood(p, y):
    ############################################################################
	# WRITEME: write your code here to complete the routine
	size=y.shape[0]
	loss = np.sum(np.multiply(y, np.log(p)))
	log_loss = -1*np.sum(loss) / size
	return log_loss

def computeCost(X, y, theta, beta): ## loss is now Bernoulli cross-entropy/log likelihood
    ############################################################################
	# WRITEME: write your code here to complete the routine
	p=regress(X,theta)
	size = X.shape[0]
	b, w = theta
	loss=log_likelihood(p,y)
	reg= (beta*np.sum(w*w))/2

	return loss+reg


def computeGrad(X, y, theta, beta):
    ############################################################################
	# WRITEME: write your code here to complete the routine (
	# NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
	p=regress(X,theta)
	# print(p)
	b, w = theta
	size=X.shape[0]
	dL_dfy = p-y # derivative w.r.t. to model output units (fy)
	dL_db = np.sum((dL_dfy),axis=0)/size# derivative w.r.t. model b
	# print(X.T.shape)
	# print(dL_dfy)
	dL_dw = np.dot(X.T,(dL_dfy/size))+beta*w# derivative w.r.t model w
	nabla = (dL_db, dL_dw) # nabla represents the full gradient
	return nabla
    ############################################################################

