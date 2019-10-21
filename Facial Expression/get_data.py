import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# Label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise','Neutral']
# df = pd.read_csv('fer2013\\fer2013\\fer2013.csv')
# # print(df[df["Usage"] == "Training"])
# df_train = df[df["Usage"] == "Training"]
# print(Label[df_train["emotion"][0]])
# # print(np.sqrt(2304))
# for image in range(len(df_train)):
	
# 	pixels=df_train["pixels"][image].split(" ")
# 	pixels = [*map(int, pixels)]
# 	# pixels = np.array(pixels)
# 	print(type(pixels))
# 	pixels = np.asmatrix(pixels)
# 	print(type(pixels),len(pixels),pixels)
# 	plt.imshow(pixels.reshape(48,48))
# 	plt.title(Label[df_train["emotion"][image]])
# 	plt.show()
	
# 	if image == 5:
# 		break
def getdata(balance_ones=True):
    # images are 48x48 = 2304 size vectors
    Y = []
    X = []
    first = True
    for line in open('fer2013\\fer2013\\fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)

    if balance_ones:
        # balance the 1 class
        X0, Y0 = X[Y!=1, :], Y[Y!=1]
        X1 = X[Y==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1]*len(X1)))

    return X, Y
# X,Y = getdata()
# print(len(X), len(X[0]),type(X))
