from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
#from keras.datasets import cifar10
from sklearn.cross_validation import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
import pickle
import h5py
import time
from keras.optimizers import SGD
def shape_array(train_x,im_row,im_col):	
	data = np.zeros((train_x.shape[0],3,im_row, im_col))
   	ds = len(train_x)
        for i in range(ds):
		for j in range(im_row):
			for k in range(im_col):
				data[i][0][j][k] = train_x[i][j][k][0]
				data[i][1][j][k] = train_x[i][j][k][1]
				data[i][2][j][k] = train_x[i][j][k][2]
	print (data.shape)
	return data

def create_model(nb_conv,img_rows,img_cols,nb_pool):
	model = Sequential()
	#(a) 
	model.add(Convolution2D(8, 11, 11,input_shape=( 3,img_rows, img_cols)))
	model.add(Activation('relu'))

	#(b)
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

	#(c) 
	model.add(Convolution2D(16, 10, 10))
	model.add(Activation('relu'))

	#(d)  
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

	#(e) 
	model.add(Convolution2D(32, 14, 14))
	model.add(Activation('relu'))

	#(f) 
	model.add(Convolution2D(64, 11, 11))
	model.add(Activation('relu'))

	#(g) 
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	#(h) 
	model.add(Convolution2D(125, 1, 1))
	model.add(Activation('relu'))
	# #(i) 
	# model.add(Convolution2D(125, 1, 1))
	# model.add(Activation('relu'))
	#(j)
	model.add(Convolution2D(2, 1, 1))
	model.add(Activation('linear'))

	model.add(BatchNormalization(axis = 1))
	model.add(Flatten())
	model.add(Activation('softmax'))

	return model

def train(nb_conv,img_rows,img_cols,nb_pool,batch_size,nb_epoch,X_train,Y_train,filepath):
	model = create_model(nb_conv,img_rows,img_cols,nb_pool)
	lr = 0.08
	momentum = 0.9
	sgd = SGD(lr=lr, momentum=momentum, nesterov=False)
	model.compile(loss='categorical_crossentropy', optimizer='sgd')
	checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
	print (X_train.shape)
	hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_split=0.25,callbacks=[checkpointer])
	print(hist.history)

def main():
	batch_size = 32
	# nb_classes = 2
	nb_epoch = 60
	# input image dimensions
	img_rows, img_cols = 128,128

	# size of pooling area for max pooling
	nb_pool = 2
	# convolution kernel size
	nb_conv = 3
	f = h5py.File("/NOBACKUP/catsanddogs.hdf5", "r")
	data_x = f["/data/x_128x128"][()]
	print (type(data_x))
	# data_x = 
	# data_x_2 = np.concatenate((data_x[0:100] ,data_x[13000:13100]), axis = 0)
	print (data_x.shape,"data_x_shape")
#	print (data_x)
	data_y = f["/data/y"][()]
	# data_y_2 = np.concatenate((data_y[0:100] , data_y[13000 :13100]), axis = 0)
#       print (data_y[0])
	data_x_2 = data_x
	data_y_2 = data_y
	# data_x_2 = shape_array(data_x_2,img_rows, img_cols)
	data_x_2 = data_x_2.transpose(0,3,1,2)
	data_x_2 = data_x_2.reshape(data_x_2.shape[0], 3,img_rows, img_cols,)
	data_x_2 = data_x_2.astype("float32")
	print("start")
	filepath = "weights_02.hdf5"
	print (data_x_2.shape,"data_x_shape")
 	print (data_y_2.shape,"data_y_shape")
	p_3 = np.random.permutation(len(data_x_2))
	x = data_x_2[p_3]
	y = data_y_2[p_3]
	train(nb_conv,img_rows,img_cols,nb_pool,batch_size,nb_epoch,x,y,filepath)
if __name__ == "__main__":
	t1 = time.time()
	main()
	t2 = time.time()
	print ("using  %s seconds" % (t2-t1))
