import numpy as np
from glob import glob
from PIL import Image
import scipy.misc
import os
import pdb
import random

class data_loader():
    def __init__(self, pr_path, gt_path):
	self.pr_path = pr_path
	self.gt_path = gt_path

	self.pr_list = glob(os.path.join(self.pr_path, "*.jpg"))
	self.gt_list = glob(os.path.join(self.gt_path, "*.jpg"))

	self.num_train = len(self.pr_list)
	self.start = 0
	self.current = 0
	random.shuffle(self.pr_list)
	random.shuffle(self.gt_list)

    def sequential_sample(self, batch_size, start=None):
        self.start = self.current
        self.current = (self.start+batch_size) % self.num_train

        pr = []
        image = np.zeros([batch_size, 80, 80, 3])
        image_real = np.zeros([batch_size, 80, 80, 3])
        if (self.start + batch_size) < self.num_train:
            for j in range(batch_size):
                i = self.start + j
                image[j,:] = scipy.misc.imread(self.pr_list[i]).astype(np.float)
                image_real[j,:] = scipy.misc.imread(self.gt_list[i]).astype(np.float)
                img_name = self.pr_list[i].split('/')[3]
                pr.append(Image.open(self.pr_list[i]))
        else:
            random.shuffle(self.pr_list)
            random.shuffle(self.gt_list)
            self.start = 0
            self.current = 0
            for j in range(batch_size):
                i = self.start + j
                image[j,:] = scipy.misc.imread(self.pr_list[i]).astype(np.float)
                image_real[j,:] = scipy.misc.imread(self.gt_list[i]).astype(np.float)
                img_name = self.pr_list[i].split('/')[3]
                pr.append(Image.open(self.pr_list[i]))

        return image, image_real, pr

    def random_sample(self, batch_size):
	idx = np.random.random_integers(0, len(self.pr_list)-1, size=(batch_size))

	pr = []
	gt = []
	image = np.zeros([batch_size, 80, 80, 3])
	for i in range(batch_size):	
	    image[i,:] = scipy.misc.imread(self.pr_list[idx[i]]).astype(np.float)
	    img_name = self.pr_list[idx[i]].split('/')[3]
	    pr.append(Image.open(self.pr_list[idx[i]]))
	    gt.append(Image.open(os.path.join(self.gt_path, img_name)))

        return image, pr, gt

    def random_single_sample(self):
	idx = np.random.random_integers(0, len(self.pr_list))-1		
	image = scipy.misc.imread(self.pr_list[idx]).astype(np.float) 

	img_name = self.pr_list[idx].split('/')[3]	

	image = image.reshape((1, 80, 80, 3))
	pr = Image.open(self.pr_list[idx])
	gt = Image.open(os.path.join(self.gt_path, img_name))

	return image, pr, gt

    def load_image(self, img_path):
	self.test_img_path = img_path
	image_pr = Image.open(img_path)
	image = self.image2pixelarray(image_pr.resize((80, 80), Image.BILINEAR))
	return image, image_pr

    def save_image(self, image, tag=None):
        if not os.path.exists('./results'):
            os.makedirs('./results')

        img_name = self.test_img_path.split('/')
        img_name = img_name[len(img_name)-1]

	if tag==None:
	    scipy.misc.imsave('./results/' + img_name, image)
	else:
	    scipy.misc.imsave('./results/' + str(tag) + '_' + img_name , image)

    def save_image_pil(self, image):
        if not os.path.exists('./results'):
            os.makedirs('./results')

	img_name = self.test_img_path.split('/')
	img_name = img_name[len(img_name)-1]
	image.save('./results/' + img_name)	

    def image2pixelarray(self, im):
        (width, height) = im.size
        pix_list = list(im.getdata())
        pix_array = np.array(pix_list).reshape((1, height, width, 3))
        return pix_array
