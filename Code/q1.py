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
	
	p = np.random.rand(len(x[:,0]),num_topics)
	p = p/np.sum(p,axis=0)
	pi = np.random.rand(num_topics)
	pi = pi/np.sum(pi,axis=0)
	return p,pi
	

def main():
	
	x = import_data()
	p,pi = initialize_em(30,x)
	print(p)
	print(pi)

main()
