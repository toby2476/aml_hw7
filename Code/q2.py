from PIL import Image
import numpy as np
import pandas as pd
import glob, os
from scipy.stats import multivariate_normal

def import_image(filename):
	im = Image.open('../Data/'+filename)
	arr = np.array(im)
	arr = np.reshape(arr,(im.width*im.height, 3))
	return [arr,im.width,im.height]


def initialize_em(c):
	mean = np.random.randint(0,255,(c,3))
	pi = (np.ones(c))/c
	return [mean,pi]
	
def e_step(mean,pi,image):
	w = np.zeros((len(image),len(pi)))
	for j in range(len(pi)):		
		distance = np.square(image/25.5-mean[j]/25.5)
		distance = (-0.5)*np.sum(distance, axis=1)
		w[:,j] = pi[j]*np.exp(distance)
	
	w = np.transpose(np.transpose(w)/np.sum(w,axis=1))
	return w

def m_step(w,image):
	mean = np.zeros((len(w[0,:]),3))
	for k in range(3):
		mean[:,k] = [(np.dot(w[:,j],image[:,k]))/sum(w[:,j]) for j in range(len(w[0,:]))] 
	
	pi = np.zeros((len(w[0,:])))
	for j in range(len(w[0,:])):
		pi[j] = sum(w[:,j])/np.sum(w)

	return [mean,pi]
	
def EM(c,image):
	mean,pi = initialize_em(c)
	convergence = False
	while(~convergence):	
		old_mean = mean
		w = e_step(mean,pi,image)
		mean,pi = m_step(w,image)
		convergence = check_convergence(old_mean,mean,1.5)
	return [w,mean,pi]

def check_convergence(old_mean,mean,threshold):
	diff = abs(old_mean - mean)
	return (np.sum(diff)/len(diff) <= threshold)
	
	
def new_image(image,w,mean,pi,width, height,filename):
	new_img = np.zeros(image.shape)
	argmax = w.argmax(axis=1)
	for i in range(len(image)):
		new_img[i,:] = mean[argmax[i]]
	new_img = np.reshape(new_img,(height,width,3))
	new_img = new_img.astype('uint8')
	im_out = Image.fromarray(new_img,'RGB')	
	im_out.save('../Output/'+filename)

def image_segmentation(input_name,output_name,n_clusters):
	image, width, height = import_image(input_name)
	w, mean, pi = EM(n_clusters,image)
	new_image(image,w,mean,pi,width,height,output_name)

def main():
	image_segmentation('RobertMixed03.jpg','RobertMixed03_10.png',10)
	image_segmentation('RobertMixed03.jpg','RobertMixed03_20.png',20)
	image_segmentation('RobertMixed03.jpg','RobertMixed03_50.png',50)
	image_segmentation('smallstrelitzia.jpg','smallstrelitzia_10.png',10)
	image_segmentation('smallstrelitzia.jpg','smallstrelitzia_20.png',20)
	image_segmentation('smallstrelitzia.jpg','smallstrelitzia_50.png',50)
	image_segmentation('smallsunset.jpg','smallsunset_10.png',10)
	image_segmentation('smallsunset.jpg','smallsunset_20.png',20)
	image_segmentation('smallsunset.jpg','smallsunset_50.png',50)

main()
