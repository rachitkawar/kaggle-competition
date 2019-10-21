import numpy as np 



#activation function
def relu(X):
	return np.maximum(X,0,X)


def softmax(X):
	expA = np.exp(X)
	
	return expA / expA.sum(axis=1,keepdims=True)


def sigmoid(X):
	return 1/(1+np.exp(-X))


def y_indicator(Y) :
	N = len(Y)
	S = len(set(Y))
	T = np.zeros((N,S))
	for i in range(N):
		T[i,Y[i]]=1
	return T

def cost(T,Y):
	return -(T*np.log(Y)).sum()

def error_rate(T,Y):
	return np.mean(T != Y)




