from PIL import Image
import numpy as np
import pandas as pd
import glob, os
from scipy.stats import multivariate_normal

def import_image(filename):
	im = Image.open('../Data/'+filename)
	arr = np.array(im)
	arr = np.reshape(arr,(im.width*im.height, 3))
	return arr


def initialize_em(c):
	mean = np.random.randint(0,255,(c,3))
	cov = [np.identity(3)*5000 for i in range(c)]
	prior = (np.ones(c))/c
	print('Means:')
	print(mean)
	return [mean,cov,prior]
	
def e_step(mean,cov,prior,image):
	print('Start E Step')
	w = np.zeros((len(image),len(prior)))
	#for i in range(len(image)):
	for j in range(len(prior)):		
		distance = np.square(image/25.5-mean[j]/25.5)
		distance = (-0.5)*np.sum(distance, axis=1)
		w[:,j] = prior[j]*np.exp(distance)
	
	w = np.transpose(np.transpose(w)/np.sum(w,axis=1))
	return w

def m_step(w,image):
	print('Start M Step')
	mean = np.zeros((len(w[0,:]),3))
	for k in range(3):
		mean[:,k] = [(np.dot(w[:,j],image[:,k]))/sum(w[:,j]) for j in range(len(w[0,:]))] 
	
	
	cov = [np.zeros((3,3)) for i in range(len(w[:,j]))]
	'''
	for k in range(len(w[0,:])):
		dist = image - mean[k]		
		for i in range(3):
			for j in range(3):		
				cov[k][i,j] = (np.dot(np.multiply(w[:,k],dist[:,i]),dist[:,j]))/sum(w[:,k])
	'''
	prior = np.zeros((len(w[0,:])))
	for j in range(len(w[0,:])):
		prior[j] = sum(w[:,j])/np.sum(w)
	
	print('Means:')
	print(mean)
	print('Prior:')
	print(prior)
	return [mean,cov,prior]
	
def EM(c,image):
	mean,cov,prior = initialize_em(c)
	num_iter = 20
	for i in range(num_iter):	
		w = e_step(mean,cov,prior,image)
		mean,cov,prior = m_step(w,image)

def main():
	image = import_image('121.jpg')
	EM(10,image)




main()
