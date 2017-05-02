import sys
from curve.filter_api import *
import random
from PIL import Image, ImageEnhance
import numpy as np
import pdb

class Curve():
    def __init__(self):
	self.state_shape = [80, 80]
        self.state_max = 255.
        self.state_min = 0.

    def step(self, action, pr, batch_size=None):
	observation = []
	for i in range(batch_size):
	    curve = self.construct_curve(action[i,:])
	    current_observation = filter_image(curve, np.asarray(pr[i]))
	    observation.append(current_observation)

	return observation

    def construct_curve(self, p):
	curve = np.asarray([[(64,64+p[0]),(127,127+p[1]), (190,190+p[2])], [(64,64+p[0]),(127,127+p[1]), (190,190+p[2])],
                        [(64,64+p[0]),(127,127+p[1]), (190,190+p[2])], [(64,64+p[0]),(127,127+p[1]), (190,190+p[2])]])


	return curve
	
    def step_test(self, action, pr):
	curve = self.construct_curve(action)
	return filter_image(curve, np.asarray(pr))

    def preprocess(self, image, action):
	# Luminance Manipulation
        brightness = ImageEnhance.Brightness(image)
        image_pro = brightness.enhance(action)

	return image_pro

    def luminance(self, state):
	# Reference
	# http://stackoverflow.com/questions/6442118/python-measuring-pixel-brightness
	L = 0.2126*np.mean(state[:,:,0])+0.7152*np.mean(state[:,:,1])+np.mean(0.0722*state[:,:,2])
	return L/255.

    def image2pixelarray(self, im):
	(width, height) = im.size
	pix_list = list(im.getdata())
	pix_array = np.array(pix_list).reshape((height, width, 3))

	return pix_array	
