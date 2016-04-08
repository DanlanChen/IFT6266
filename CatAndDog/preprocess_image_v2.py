from pathlib import Path
import h5py
from glob import glob
from PIL import Image
import numpy as np
from numpy import array
#train_file_path = '/NOBACKUP/cats_dog_kaggle_image_data/train/'
resize_train_path = '/NOBACKUP/cats_dog_kaggle_image_data/resize_train/'
flip_train_path = '/NOBACKUP/cats_dog_kaggle_image_data/flip_train/'
rotate_train_path = '/NOBACKUP/cats_dog_kaggle_image_data/rotate_train/'
#train_files = glob(train_file_path + '*.jpg')
flip_files = glob(flip_train_path + '*.jpg')
resize_files = glob(resize_train_path +'*.jpg')
rotate_files = glob(rotate_train_path +'*.jpg')

data_x=[]
data_y = []

def load_h5(filepath):
    print "h5py : loading data from", filepath
    h5f = h5py.File(filepath,'r')
    params = []
    for key in h5f.keys():
        params.append(h5f[key][()])
    h5f.close()
    return params
def main1():
	for i,f in enumerate(rotate_files):
			p = Path(f)
			name = p.name
		        #print name
			if name[9] == 'c' or name[10] == 'c':
				y = np.array([1,0], dtype = 'float32')
		        else:
				y = np.array([0,1], dtype = 'float32')
			img = Image.open(f)
			rotate = array(img)
			data_x.append(rotate)
			data_y.append(y)
	file = '/NOBACKUP/cats_dog_kaggle_image_data/rotate_image.hdf5'
	hp = h5py.File(file,'w')
	hp.create_dataset('rotate_x',data = np.array(data_x,dtype = 'float32'))
	hp.create_dataset('rotate_y',data = np.array(data_y,dtype = 'float32'))
	hp.close()
def flip():
	for i,f in enumerate(flip_files):

			p = Path(f)

		        print name
			if name[5] == 'c':
				y = np.array([1,0], dtype = 'float32')
		        else:
				y = np.array([0,1], dtype = 'float32')
			img = Image.open(f)
			rotate = array(img)
			data_x.append(rotate)
			data_y.append(y)
	file = '/NOBACKUP/cats_dog_kaggle_image_data/flip_image.hdf5'
	hp = h5py.File(file,'w')
	hp.create_dataset('flip_x',data = np.array(data_x,dtype = 'float32'))
	hp.create_dataset('flip_y',data = np.array(data_y,dtype = 'float32'))
	hp.close()
	"""
	print "h5py : loading data from", file
	h5f = h5py.File(file,'r')
	print h5f['rotate_x'][()].shape
	print h5f['rotate_y'][()]
        """
def resize():
	for i,f in enumerate(resize_files):

			p = Path(f)
			name = p.name

			if name[7] == 'c':
				y = np.array([1,0], dtype = 'float32')
		        else:
				y = np.array([0,1], dtype = 'float32')
			img = Image.open(f)
			rotate = array(img)
			data_x.append(rotate)
			data_y.append(y)
	file = '/NOBACKUP/cats_dog_kaggle_image_data/resize_image.hdf5'
	hp = h5py.File(file,'w')
	hp.create_dataset('resize_x',data = np.array(data_x,dtype = 'float32'))
	hp.create_dataset('resize_y',data = np.array(data_y,dtype = 'float32'))
	hp.close()
	
	print "h5py : loading data from", file
	h5f = h5py.File(file,'r')
	print h5f['resize_x'][()].shape
	print h5f['resize_y'][()]
def merge():
	file1 = '/NOBACKUP/cats_dog_kaggle_image_data/rotate_image.hdf5'
	file2 = '/NOBACKUP/cats_dog_kaggle_image_data/resize_image.hdf5'
	file3 = '/NOBACKUP/cats_dog_kaggle_image_data/flip_image.hdf5'
	file4 = '/NOBACKUP/cats_dog_kaggle_image_data/whole_image.hdf5'
        [resize_x, resize_y] = load_h5(file2)
        print resize_x.shape,resize_y.shape
        [rotate_x, rotate_y] = load_h5(file1)
        print rotate_x.shape, rotate_y.shape
        [flip_x, flip_y] = load_h5(file3)
        print flip_x.shape, flip_y.shape
        data_x = np.concatenate((resize_x,rotate_x, flip_x),axis =0)
        data_y = np.concatenate((resize_y,rotate_y,flip_y),axis =0)
        print data_x.shape, data_y.shape
        hp = h5py.File(file4,'w')
        hp.create_dataset("whole_x",data = np.array(data_x, dtype = "float32"))
        hp.create_dataset("whole_y",data = np.array(data_y, dtype = "float32"))
        hp.close()

merge()
