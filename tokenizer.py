import os 
import numpy as np 
import json
import pdb 

import math
import nltk , json 

from joblib import Parallel, delayed
import multiprocessing
import pickle 
from keras.preprocessing.text import Tokenizer

output_path = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/tpot_multi_data/'



output_files = [x for x in os.listdir (output_path) if x.endswith('.npy')]


def load_pretrained_glove():
    print("Loading GloVe model, this can take some time...")
    glv_vector = {}
    f = open('/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/glove/glove.840B.300d.txt', encoding='utf-8')

    for line in f:
        values = line.split()
        word = values[0]

        try:
            coefs = np.asarray(values[1:], dtype='float')
            glv_vector[word] = coefs
        except ValueError:
            continue
    f.close()
    print("Completed loading pretrained GloVe model.")
    return glv_vector

op= 'project'
if op == 'build_corpus':
	sentences = []
	def read_data (ii, val):

		data = np.load (os.path.join(output_path,val),allow_pickle=True).item()
		return data['text']
		
	sentences  = Parallel(n_jobs=12)(delayed(read_data)(ii,val) for ii, val in enumerate(output_files))	
	sentences = np.concatenate (sentences)
	sentences = np.concatenate (sentences)
	
	np.save ('corpus.npy', sentences)
elif op == 'dict':
	words = []
	sentences = np.load ('corpus.npy',allow_pickle=True)

	for x in sentences:
		words.extend (x.split(' '))
	unique_words = np.unique (words)

	
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(sentences)

	sequence= tokenizer.texts_to_sequences(sentences)

	
	word_index = tokenizer.word_index
	num_unique_words= len (word_index)
	inv_word_index = {v: k for k, v in word_index.items()}	

	with open('wordmap.json', 'w') as f:
		json.dump(word_index, f)
	with open('inv_wordmap.json', 'w') as f:
		json.dump(inv_word_index, f)

	# saving
	with open('tokenizer.pickle', 'wb') as handle:
		pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	glove_dir= '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/glove'

	glv_vector = np.load(os.path.join(glove_dir,'glv.npy'),allow_pickle=True).item()

	word_vector_length =300 
	glv_embedding_matrix = np.zeros((num_unique_words+1, word_vector_length))	
	for j in range(1, num_unique_words+1):
		try:
			glv_embedding_matrix[j] = glv_vector[inv_word_index[j]]
		except KeyError:
			glv_embedding_matrix[j] = np.random.randn(word_vector_length)/200
	np.ndarray.dump(glv_embedding_matrix, open(output_path + 'glv_embedding_matrix', 'wb'))

elif op == 'project':


	def map_words (text):
		return np.array([word_index[k] for k  in text])

	with open('wordmap.json') as f:
		word_index=json.load(f)

	with open('inv_wordmap.json') as f:
		inv_word_index=json.load(f)

	with open('tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)

	def read_data (idx, val):
		print (idx, val.split('.')[0].split('_')[-1])
	
		data = np.load (os.path.join(output_path,val),allow_pickle=True).item()
		total_seq= []
		for dat in data['text']:
			seq_list = []
			
			seq_list = tokenizer.texts_to_sequences (dat)
			
			total_seq.append (seq_list)
		data['sequence'] = total_seq
		#pdb.set_trace()
		np.save(output_path+'split_' + val.split('.')[0].split('_')[-1] +'.npy', data)

	sentences  = Parallel(n_jobs=12)(delayed(read_data)(ii,val) for ii, val in enumerate(sorted(output_files)))
	print ('Done')

'''
token_child = []
token_mother = [] 	 
for val_c, val_p in zip(ling_child,  ling_mother):

	c_text= val_c ['text']
	p_text= val_p ['text']
	c_token = [map_words (w) for w in  c_text]
	p_token = [map_words (w) for w in  p_text]
	token_child.append (c_token)
	token_mother.append (p_token)

data['token_child']  = np.array(token_child)
data['token_mother'] = np.array(token_mother)

print ("Done")
'''