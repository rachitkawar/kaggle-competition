import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from get_data import getdata
from sklearn.utils import shuffle
from function import getdata,y_indicator, error_rate, init_weight_and_bias

class HiddenLayer(object):
	"""docstring for HiddenLayer"""
	def __init__(self, M1,M2,an_id):
		self.M1 = M1
		self.M2 = M2
		self.id = an_id
		W,b = init_weights(M1,M2)
		self.W = tf.Variable(W.astype(tf.float32))
		self.b = tf.Variable(b.astype(tf.float32))
		self.params = [self.W,self.b]

	def forward(self,X):
		return tf.nn.relu(tf.matmul(X , self.W)+self.b)
	def init_weight_and_bias(M1, M2):
    	W = np.random.randn(M1, M2) / np.sqrt(M1)
    	b = np.zeros(M2)
    	return W.astype(np.float32), b.astype(np.float32)

		
class anntf(object):
	"""docstring for anntf"""
	def __init__(self, size_ofhidden_layer):
		self.size_of_hidden_layer = size_ofhidden_layer


	def fit(self,X,Y,learning_rate = 1e-6 , show_fig = False , batch_size = 100 ,decay = 0.990 , epochs = 10 , mu = 0.99):
		X,Y = shuffle(X,Y)
		K = len(set(Y))
		X = X.astype(np.float32)
		T = y_indicator(Y).astype(np.float32)
		Xvalid , Yvalid = X[-1000:],Y[-1000:]
		X ,Y = X[:-1000] , Y[-1000:]
		Yvalid_flat = np.argmax(Yvalid, axis=1)
		N,D = X.shape
		self.hidden_layer = []
		M1 = D
		count = 0
		for M2 in self.size_of_hidden_layer:
			h = HiddenLayer(M1,M2,count)
			self.hidden_layer.sppend(h)
			M1 = M2
			count+=1
		W,b = init_weights(M1,K)
		self.W = tf.Variable(W.astype(tf.float32))
		self.b = tf.Variable(b.astype(tf.float32))

		self.params = [self.W,self.b]
		for h in hidden_layer:
			self.params += h.params

		tfX = tf.placeholder(tf.float32 , shape = (None,D), name = "X")
		tfY = tf.placeholder(tf.float32 , shape = (None , K) , name = "Y")
		logits = self.forward(tfX)

		cost  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
			logits = logits,
			labels = tfY))
		prediction  = self.predict(tfX)
		train_op = tf.trainRMSPropOptimizer(learning_rate,decay = decay , momentum = mu).minimize(cost)
		n_batches = N// batch_size
		costs =  []
		init = tf.global_variables_initializer()
		with tf.Session() as sees:
			sees.run(init)
			for i in range(epochs):
				X,Y = shuffle(X,Y)
				for j in range(n_batches):
					X_batch = X[j*batch_size:(j+1)*batch_size]
					Y_batch = Y[j*batch_size:(j+1)*batch_size]
					sees.run(train_op , feed_dict = {tfX: X_batch , tfY: Y_batch})
					
					if j%20 == 0:
						c = session.run(cost, feed_dict={tfX: Xvalid, tfT: Yvalid})
						costs.append(c)
						p = sees.run(prediction,feed_dict= {tfX:Xvalid , tfY:Yvalid})
						e = error_rate(Yvalid_flat,p)
			if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return tf.matmul(Z, self.W) + self.b

    def predict(self, X):
        act = self.forward(X)
        return tf.argmax(act, 1)
    def init_weight_and_bias(M1, M2):
    	W = np.random.randn(M1, M2) / np.sqrt(M1)
    	b = np.zeros(M2)
    	return W.astype(np.float32), b.astype(np.float32)



def main():
    X, Y = getData()
    # X, Y = getBinaryData()
    model = ANN([2000, 1000, 500])
    model.fit(X, Y, show_fig=True)

if __name__ == '__main__':
    main()

		





