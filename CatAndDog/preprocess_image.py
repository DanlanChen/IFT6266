from PIL import Image
from glob import glob
from pathlib import Path
import PIL
from numpy import array
def resize_image(img,w,h,outputfile):
	# load the image and show it
	arr = array(img)
	print arr.shape
	# print arr
	# img2 = Image.fromarray(img,'RGB')
	img2 = img.resize((w, h), PIL.Image.ANTIALIAS)
	img2.save(outputfile)

	# cv2.imshow("resized", resized)
	# cv2.waitKey(0)
def flip_image(img,outputfile):
	img2 = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
	img2.save(outputfile)
       
def rotate_image(img, degree1,degree2, outputfile1,outputfile2):
	h,w = img.size[0], img.size[1]
        box = (int(0.08 * w), int(0.05 * h), int(0.87* w), int(0.95 * h))
        print box
        img2 = img.rotate(degree1)
        img2 = img2.crop(box)
	img2.save(outputfile1)
	img3 = img.rotate(degree2)
        img3 = img3.crop(box)
	img3.save(outputfile2)
	print img2.size
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
	for f in train_files:
		p = Path(f)
		resize_file = resize_train_path + "resize_" + p.name 
		flip_file = flip_train_path + "flip_" + p.name
		rotate_fil_1 = rotate_train_path + "rotate_" + str(d1) + "_" + p.name
		rotate_fil_2 = rotate_train_path +"rotate_" + str(d2) + "_" + p.name
		print resize_file
		print flip_file
		print rotate_fil_1
		print rotate_fil_2
                img = Image.open(f)
		resize_image(img,w,h,resize_file)
		flip_image(img,flip_file)
		rotate_image(img,d1,d2,rotate_fil_1,rotate_fil_2)

		break
if __name__ == "__main__":
	main()
