import pandas as pd 
import numpy as np 
import os 
import pdb 

from joblib import Parallel, delayed
import multiprocessing


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import random 
#data_dir = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/segment_features/'

#fold_dir = '../../LIFE_Codes/kfold_data/'


def combine_splits(data_dir):

	data_files = sorted([x for x in os.listdir(data_dir) if x.endswith('.npy') and x.startswith('split_') and 'DATA' not in x])

	full_data = dict()
	

	def read_data (i, dat_file):
		inp_ = np.load (os.path.join(data_dir, dat_file), allow_pickle=True).item()
		return inp_ 

		
	data_  = Parallel(n_jobs=12)(delayed(read_data)(ii,val) for ii, val in enumerate(sorted(data_files)))
	
	for i,data in enumerate(data_):
		
		if i == 0: # Populate the dictonary
			for keys, 	values in data.items():
				if not isinstance (values, list):
					full_data[keys] = values.tolist()
				else:
					full_data [keys] = values
		else:
			for keys,values in data.items():
				full_data[keys].extend (values)

		#print (len(data['turn_filename']))

	return full_data

def get_k_fold ( list_ , n_folds=2):

	kf = KFold (n_splits= n_folds)
	kf.get_n_splits (list_)

	tr_indices , vl_indices , ts_indices =[],[],[]
	for train_index, test_index in kf.split(list_):
		
		valid_index= random.sample (list(train_index),int(len (train_index) * 0.15))
		
		tr_id = np.array(list(set(train_index) - set (valid_index)))
		
		vl_id = np.array(valid_index)
		ts_id = test_index.copy()
		
		tr_indices.append (tr_id)
		vl_indices.append (vl_id)
		ts_indices.append (ts_id)
	return tr_indices, vl_indices, ts_indices
def get_split_indices (fold_dir, full_data, fold_no=0):

	folds= np.array(os.listdir(fold_dir)) 
	fold_file = folds[folds  == 'split_' + str(fold_no)+'.npy'][fold_no]
	

	data= np.load(os.path.join(fold_dir, fold_file ), allow_pickle=True).item()
	x_train,x_valid,x_test=data['x_train'],data['x_valid'],data['x_test']
	
	#keys = np.array(full_data['turn_filename']) 
	keys= np.array(full_data['family_id'])
	tr_indices = np.array([np.where(keys==x)[0] for x in x_train if x in keys]).flatten()

	vl_indices = np.array([np.where(keys==x)[0] for x in x_valid if x in keys]).flatten()

	ts_indices = np.array([np.where(keys==x)[0] for x in x_test if x in keys]).flatten()

	return tr_indices, vl_indices, ts_indices

'''
def main ():

	full_data = combine_splits(data_dir)

	
	#----Put this line in a  loop for k-fold experiment ------#
	tr_indices, vl_indices, ts_indices= get_split_indices(fold_dir, full_data, fold_no=2)


if __name__== "__main__":
	main()

'''
