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
def flip_image(img):
	img2 = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
#	img2.save(outputfile)
        return img2
def rotate_image(img, degree1,degree2):
	h,w = img.size[0], img.size[1]
        box = (int(0.08 * w), int(0.05 * h), int(0.87* w), int(0.95 * h))
        #print box
        img2 = img.rotate(degree1)
        img2 = img2.crop(box)
	#img2.save(outputfile1)
	img3 = img.rotate(degree2)
        img3 = img3.crop(box)
	#img3.save(outputfile2)
	#print img2.size
        return img2, img3
	# print Image.fromarray(img2,'RGB').shape()
def main():
	train_file_path = '/NOBACKUP/cats_dog_kaggle_image_data/train/'
	resize_train_path = '/NOBACKUP/cats_dog_kaggle_image_data/resize_train/'
	flip_train_path = '/NOBACKUP/cats_dog_kaggle_image_data/flip_train/'
	rotate_train_path = '/NOBACKUP/cats_dog_kaggle_image_data/rotate_train/'
	train_files = glob(train_file_path + '*.jpg')
	# resize_files = glob(resize_files + '.jpg')
	w,h = 128,128
	d1 = -7
	d2 = 7
        data = {"orig":[],"flip":[],"r1":[],"r2":[],"all":[]}
	for i,f in enumerate(train_files):
                if i % 1000 == 0:
                    print "processed " + str(i) +"images"
		p = Path(f)
		resize_file = resize_train_path + "resize_" + p.name 
		flip_file = flip_train_path + "flip_" + p.name
		rotate_fil_1 = rotate_train_path + "rotate_" + str(d1) + "_" + p.name
		rotate_fil_2 = rotate_train_path +"rotate_" + str(d2) + "_" + p.name
		#print resize_file
		#print flip_file
		#print rotate_fil_1
		#print rotate_fil_2
	        img = Image.open(f)
	        orig = array(img)
		resize = resize_image(img,w,h,resize_file)
	        resize = array(resize)
		flip = flip_image(img)
	        flip = resize_image(flip,w,h,flip_file)
		flip = array(flip)
		r1, r2 = rotate_image(img,d1,d2)
	        r1 = resize_image(r1,w,h,rotate_fil_1)
	        r2 = resize_image(r2,w,h,rotate_fil_2)
	        r1, r2 = array(r1), array(r2)
	        #print resize.shape,flip.shape,r1.shape, r2.shape
	        data["orig"].append(resize)
	        data["flip"].append(flip)
	        data["r1"].append(r1)
	        data["r2"].append(r2)
	data["orig"] = np.asarray(data["orig"])
        data["flip"] = np.asarray(data["flip"])
        data["r1"] = np.asarray(data["r1"])
        data["r2"] = np.asarray(data["r2"])
        print data["orig"].shape
        print data["flip"].shape
        print data["r1"].shape
        print data["r2"].shape
        data["all"] = np.concatenate((data["orig"], data["flip"] ,data["r1"] , data["r2"]), axis =0)
        print data["all"].shape
        filepath = '/NOBACKUP/cats_dog_kaggle_image_data/rotate_flip_resize.hdf5'
        h5f = h5py.File(filepath, 'w')
 
        for k,v in data.iteritems():
            h5f.create_dataset(k,data = np.array(v, dtype = 'float32'))
        h5f.close()
       
if __name__ == "__main__":
        """
        filepath = '/NOBACKUP/cats_dog_kaggle_image_data/rotate_flip_resize.hdf5'
        print "h5py : loading data from", filepath
	h5f = h5py.File(filepath,'r')
        print h5f.keys()
	params = []
	for key in h5f.keys():
		params.append(h5f[key][:])
	h5f.close()
	print params[0].shape
        """
	main()
