
import proc_image
import Image
import glob
import os


if __name__ == "__main__":
	cut = proc_image.ProcImage()
	image = Image.open('testcrop.png')
	print image.size
	list_dirs = os.walk('./yalefaces/')
	for root, dirs, files in list_dirs:
		for addr in dirs:
			print addr
	w = []
	#w = cut.whiten(image, 96, 96)
	cut.convertImgTobin('testcrop.png', 'test.bin', crop_ul = 0)
