import numpy as np
import pandas as pd
import glob, os

def import_data():

	vocab = pd.read_csv('../Data/vocab.nips.txt',header=None, names = ['Word'])
	docword = pd.read_csv('../Data/docword.nips.txt',sep=' ',skiprows = 3,header=None,names = ['DocID','WordID','Count'])

	total_words = len(docword)
	total_vocab = max(docword['WordID'])
	total_docs = max(docword['DocID'])

	x = np.zeros((total_docs,total_vocab))
	for i in range(1,total_docs+1):
		row = [0]*total_vocab
		filtered = docword[docword['DocID']==i]
		for j,r in filtered.iterrows():
			row[r['WordID']-1] = r['Count']
		x[i-1,:] = row

	return x


def initialize_em(num_topics,x):
	
	p = np.random.rand(len(x[0,:]),num_topics)
	p = p/np.transpose(np.repeat(np.expand_dims(np.sum(p,axis=0),axis=1),len(x[0,:]),axis=1))
	pi = np.ones((num_topics))/num_topics
	return p,pi

def e_step(x,p,pi):
	print('start e step')
	p[p<1e-300] = 1e-300
	w = np.dot(x,np.log(p))+np.log(pi)
	max_val = np.amax(w, axis=1)
	w = np.exp(w - max_val[:,None])
	w = w/((np.sum(w,axis=1))[:,None])
	return w

def m_step(x,w):
	print('start m step')
	p = np.matmul(np.transpose(x),w)
	p = p/np.transpose(np.repeat(np.expand_dims(np.sum(p,axis=0),axis=1),len(x[0,:]),axis=1))

	pi = np.sum(w,axis=0)/len(w)
	return p, pi	

def EM(num_topics,x):
	p,pi = initialize_em(num_topics,x)
	for i in range(10):
		w = e_step(x,p,pi)
		p,pi = m_step(x,w)
	return [w,p,pi]

def main():
	
	x = import_data()
	w,p,pi = EM(30,x)
	print(p)
	print(pi)
	
	

main()
