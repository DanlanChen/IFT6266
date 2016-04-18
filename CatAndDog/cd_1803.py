from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)
from keras.models import Sequential,model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
#from keras.datasets import cifar10
from sklearn.cross_validation import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler
import pickle,os
import h5py,csv
import time,json
from keras.optimizers import Adam,SGD,RMSprop
from keras.regularizers import l1l2,l2,activity_l2
from itertools import izip
# INCREASE CONV NET LAYERS
def load_h5(filepath):
    print ("h5py : loading data from", filepath)
    h5f = h5py.File(filepath,'r')
    params = []
    for key in h5f.keys():
        params.append(h5f[key][()])
    h5f.close()
    return params

def create_model(nb_conv,img_rows,img_cols,nb_pool,file):
	model = Sequential()
	#(a) 
	model.add(Convolution2D(30, 4, 4,W_regularizer=l1l2(l1 =0.01 ,l2 =0.01),input_shape=( 3,img_rows, img_cols)))
	model.add(Activation('relu'))
	# model.add(Dropout(0.8))

	#(b)
	# model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

	#(c) 
	model.add(Convolution2D(60, 3, 3,W_regularizer=l1l2(l1 =0.01 ,l2 =0.01)))
	model.add(Activation('relu'))

	#(d)  
	model.add(MaxPooling2D(pool_size=(3, 3)))
	model.add(Dropout(0.5))

	#(e) 
	model.add(Convolution2D(128, 2, 2,W_regularizer=l1l2(l1 =0.01 ,l2 =0.01)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	#(f) 
	model.add(Convolution2D(90, 2, 2,W_regularizer=l1l2(l1 =0.01 ,l2 =0.01)))
	model.add(Activation('relu'))

	model.add(Convolution2D(120, 2, 2,W_regularizer=l1l2(l1 =0.01 ,l2 =0.01)))
	model.add(Activation('relu'))

	#(g) 
	model.add(MaxPooling2D(pool_size=(3, 3)))

	model.add(Convolution2D(150, 3, 3,W_regularizer=l1l2(l1 =0.01 ,l2 =0.01)))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Convolution2D(250, 2, 2,W_regularizer=l1l2(l1 =0.01 ,l2 =0.01)))
	model.add(Activation('relu'))


	#(h) 
	model.add(Convolution2D(256, 1, 1,W_regularizer=l1l2(l1 =0.01 ,l2 =0.01)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	#(i) 
	model.add(Convolution2D(256, 1, 1,W_regularizer=l1l2(l1 =0.01 ,l2 =0.01)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	#(j)
	model.add(Convolution2D(2, 1, 1,W_regularizer=l1l2(l1 =0.01 ,l2 =0.01)))
	model.add(Activation('linear'))

	model.add(BatchNormalization(axis = 1))
	model.add(Flatten())
	model.add(Activation('softmax'))
	model_json = model.to_json()
	with open(file,'w') as outfile:
		json.dump(model_json,outfile)
	return model
# def scheduler(ep,model):
# 	if epoch == ep:
# 		model.lr.set_value(.001)
# 	return model.lr.get_value()
def train(nb_conv,img_rows,img_cols,nb_pool,batch_size,nb_epoch,X_train,Y_train,filepath,f1,optimizer,f2):
	model = create_model(nb_conv,img_rows,img_cols,nb_pool,f1)
	if os.path.exists(filepath):
		model.load_weights(filepath)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)
	checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
	early_stop = EarlyStopping(monitor = 'val_loss', patience = 5, mode = 'min')
	print (X_train.shape)
	hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_split=0.25,callbacks=[checkpointer,early_stop])
	with open(f2,'w') as outputfile:
		json.dump(hist.history,outputfile)
	print(hist.history)
	# print(hist.history)
def predict(x,f1,f2,f3,optimizer):
    with open(f1) as in_file:
        in_file =json.load(in_file)
    model = model_from_json(in_file)
    model.load_weights(f2)
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    predictions = model.predict_classes(x, batch_size = 129,verbose = 1)
    print (predictions)
    
    #predictions = list(np.argmax(predictions,axis = 1))
    x= list(np.arange(1, 1 + len(predictions)))
    with open(f3,'wb') as f:
        writer = csv.writer(f)
        writer.writerow( [ 'id', 'label' ] )
        writer.writerows(izip(x,predictions))
    print ('file saved' + str(f3))
def main():
	batch_size = 32
	nb_classes = 2
	nb_epoch = 10
	# input image dimensions
	img_rows, img_cols = 128,128

	# size of pooling area for max pooling
	nb_pool = 2
	# convolution kernel size
	nb_conv = 3
	#optimizer
	adam = Adam(lr = 0.008)
	lr = 0.01
	momentum = 0.9
	sgd = SGD(lr=lr, momentum=momentum, nesterov=False)
	rmsprop = RMSprop(lr = 0.005, rho = 0.9, episilon = 1e-06)
	rmsprop2 = RMSprop(lr = 0.001)
	f = h5py.File("/NOBACKUP/catsanddogs.hdf5", "r")

	data_x = f["/data/x_128x128"][()]
	data_x = data_x.transpose(0,3,1,2)
	data_x = data_x.reshape(data_x.shape[0], 3,img_rows, img_cols,)
	data_x = data_x.astype("float32")
	# data_x = data_x[0:100]
	print (data_x.shape,"data_x_shape")
#	print (data_x)
	data_y = f["/data/y"][()]
	p_3 = np.random.permutation(len(data_x))
	# p_3 = np.random.permutation(50)
	x = data_x[p_3]
	y = data_y[p_3]
	f1,f2,f3,f4,f5 = "model_04_16_5.json", "weights_04_16_5.hdf5","prediction_model_04_16_5.csv","05_model_train_1.json","05_model_train_2.json"  
	train(nb_conv,img_rows,img_cols,nb_pool,batch_size,nb_epoch,x,y,f2,f1,rmsprop,f4)
	train(nb_conv,img_rows,img_cols,nb_pool,batch_size,nb_epoch,x,y,f2,f1,rmsprop2,f5)

	test_file = '/NOBACKUP/cats_dog_kaggle_image_data/test_resize.hdf5'
	f = h5py.File(test_file,'r')
	x_test = f["test_resize"][()]
	#[x_test] = load_h5(test_file)
	#print (x_test[1].shape)
	#print (x_test[1])
	#im = Image.fromarray(np.array(x_test[1],dtype = 'unint8'),'RGB')
	#im.save("0_test.jpg")
	x_test = x_test.transpose(0,3,1,2)
	x_test = x_test.reshape(x_test.shape[0], 3,img_rows, img_cols,)
	x_test = np.array(x_test, dtype = "float32")
	print (x_test.shape)
	      
	predict(x_test,f1,f2,f3,rmsprop)
if __name__ == "__main__":
	t1 = time.time()
	main()
	t2 = time.time()
	print ("using  %s seconds" % (t2-t1))
