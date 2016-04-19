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
def load_h5(filepath):
    print ("h5py : loading data from", filepath)
    h5f = h5py.File(filepath,'r')
    params = []
    for key in h5f.keys():
        params.append(h5f[key][()])
    h5f.close()
    return params

def predict(x,index,f1,f2,f3,optimizer):
    with open(f1) as in_file:
        in_file =json.load(in_file)
    model = model_from_json(in_file)
    model.load_weights(f2)
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    predictions = model.predict_classes(x, batch_size = 129,verbose = 1)
    print (predictions)
    
    #predictions = list(np.argmax(predictions,axis = 1))
    # x= list(np.arange(1, 1 + len(predictions)))
    with open(f3,'wb') as f:
        writer = csv.writer(f)
        writer.writerow( [ 'id', 'label' ] )
        writer.writerows(izip(index,predictions))
    print ('file saved ' + str(f3))
f1,f2,f3= "../model_04_18_7.json", "../weights_04_18_7.hdf5","prediction_model_04_18_7.csv"
lr = 0.01
momentum = 0.9
sgd = SGD(lr=lr, momentum=momentum, nesterov=False)
rmsprop = RMSprop(lr = 0.005, rho = 0.9, episilon = 1e-06)
rmsprop2 = RMSprop(lr = 0.001)

test_file = '/NOBACKUP/cats_dog_kaggle_image_data/resize_test.hdf5'
# f = h5py.File(test_file,'r')
[index,x_test ] =load_h5(test_file)
# x_test = f["test_resize"][()]
#[x_test] = load_h5(test_file)
#print (x_test[1].shape)
#print (x_test[1])
#im = Image.fromarray(np.array(x_test[1],dtype = 'unint8'),'RGB')
#im.save("0_test.jpg")
x_test = np.array(x_test, dtype = "float32")
print (x_test.shape)
print (x_test[0])
x_test = x_test.transpose(0,3,1,2)
x_test = x_test.reshape(x_test.shape[0], 3,128, 128,)
x_test = np.array(x_test, dtype = "float32")
print (x_test.shape)
      
predict(x_test,index,f1,f2,f3,sgd)