import numpy as np
import matplotlib.pyplot as plt
from get_data import getdata 
from function import relu , y_indicator , cost , error_rate , sigmoid , softmax
from sklearn.utils import shuffle
# np.warnings.filterwarnings('ignore')
class ann(object):
	"""docstring for ann"""
	def __init__(self,M):
		
		self.M = M
	def forward(self,X):
		Z = np.tanh(X.dot(self.W1)+self.b1)
		return softmax(Z.dot(self.W2)+self.b2),Z
	def fit(self,X,Y,learning_rate = 1e-6,epochs = 10000 , showfig = False):
		X,Y = shuffle(X,Y)
		X = X[:-1000]
		Y = Y[:-1000]
		X_test = X[-1000:]
		Y_test = Y[-1000:]

		N,D = X.shape

		K = len(set(Y))

		T = y_indicator(Y)
		self.W1 = np.random.randn(D,self.M) / np.sqrt(D)
		self.b1 = np.zeros(self.M)
		self.W2 = np.random.randn(self.M,K) / np.sqrt(self.M)
		self.b2 = np.zeros(K)
		
		costs = []
		best_val_err=1

		for i in range(epochs):
			pY , Z = self.forward(X)

			
			self.W2 += learning_rate*(Z.T.dot(T-pY))
			self.b2 += learning_rate*(T-pY).sum(axis=0)
			dz = (T-pY).dot(self.W2.T)*(1-Z*Z)
			self.W1 += learning_rate*(X.T.dot(dz))
			self.b1 += learning_rate*(dz).sum(axis=0)


			if i%20 ==0:
				pYvalid, _ = self.forward(X_test)
				pred = np.argmax(pYvalid,axis=1)
				T2 = y_indicator(Y_test)
				c = cost(T2,pYvalid)
				costs.append(c)
				er = error_rate(Y_test,pred)
				print("i:", i, "cost:", c, "error:", er)
				if er < best_val_err:
					best_val_err=er
		print("best validation error :",best_val_err)
		plt.plot(costs)
		plt.show()


	def predict(self,X):
		pY,_ = self.forward(X)


		return np.argmax(pY,axis=1)


	def score(self,X,Y):
		prediction = self.predict(X)
		return 1 - error_rate(Y, prediction)


def main():
	X,Y = getdata()
	print(len(X))
	model = ann(200)
	model.fit(X,Y)
	print(model.score(X,Y))



if __name__ == "__main__":
	main()



