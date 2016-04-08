from PIL import Image
from glob import glob
from pathlib import Path
import PIL
from numpy import array
import h5py
import numpy as np
def resize_image(img,w,h,outputfile):
	# load the image and show it
	arr = array(img)
	#print arr.shape
	# print arr
	# img2 = Image.fromarray(img,'RGB')
	img2 = img.resize((w, h), PIL.Image.ANTIALIAS)
	img2.save(outputfile)
        return img2
	# cv2.imshow("resized", resized)
	# cv2.waitKey(0)
def load_h5f(filepath):
    print "h5py : loading data from", filepath
    h5f = h5py.File(filepath,'r')
    params = []
    for key in h5f.keys():
        params.append(h5f[key][:])
    h5f.close()
    return params
def main():
	test_file_path = '/NOBACKUP/cats_dog_kaggle_image_data/test1/'
        test_files = glob(test_file_path + '*.jpg')
        test_resize_path = '/NOBACKUP/cats_dog_kaggle_image_data/test_resize/'
        w,h = 128,128
        test_files = glob(test_file_path + '*.jpg')
        data = []
        for i,f in enumerate(test_files):
            if i % 1000 == 0:
               print "processed " + str(i) +"test_image" 
	    p = Path(f)
	    outputfile = test_resize_path + "resize_test" + p.name 
	    img = Image.open(f)
	    img2 = resize_image(img,w,h,outputfile)
	    img2 = array(img2)
	    data.append(img2)
	filepath ='/NOBACKUP/cats_dog_kaggle_image_data/test_resize.hdf5'
	h5p = h5py.File(filepath,'w')
	h5p.create_dataset("test_resize",data = np.array(data,dtype = "float32"))
if __name__ == "__main__":
	main()
        """
        filepath = '/NOBACKUP/cats_dog_kaggle_image_data/test_resize.hdf5'
        params = load_h5f(filepath)
        print params[0].shape
        """
