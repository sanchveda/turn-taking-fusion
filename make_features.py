import numpy as np 
import os , csv 
import pdb 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd 
#import xlsxwriter
from scipy.io import loadmat
from scipy import signal 
from scipy.stats import iqr 
import statsmodels.api as sm
import statsmodels.formula.api as smf

from joblib import Parallel, delayed
import multiprocessing

import seaborn as sns 
from itertools import combinations 


from sklearn import preprocessing

feature_path = '/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/pipeline_output/landmarks/'
au_occ_path = '/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/pipeline_output/AU_input/formatted/occurrence/'
au_int_path = '/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/pipeline_output/AU_input/formatted/intensity/'

#gaze_path = '/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/Sanchayan/DAIC-WOZ/gaze/'
output_path = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/tpot_multi_data/'

feature_files = np.array (sorted ([x for x in os.listdir (feature_path) if x.endswith('mat')]))
au_occ_files = np.array (sorted ([x for x in os.listdir (au_occ_path) if x.endswith('mat')]))
#gaze_files = np.array (sorted ([x for x in os.listdir (gaze_path) if x.endswith('txt')]))

audio_path = '/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/audio/audio_processed/opensmile_all/PSI/'

covarep_files = np.array(sorted([x for x in os.listdir (audio_path) if x.endswith('csv')]))
def handle_opensmile_df (opensmile_dfs, start,end,skip=25):

	res_vector= np.empty(0)
	opensmile_lld_df,opensmile_csv_df,prosody_df,vad_df = opensmile_dfs

	
	csv_vector= opensmile_csv_df[start:end:skip]
	lld_vector= opensmile_lld_df[start:end:skip]
	prosody_vector= prosody_df [start:end:skip]
	vad_vector    = vad_df [start:end:skip]

	
	if csv_vector.size :#and lld_vector.size and prosody_vector.size and vad_vector.size:
	
		med_csv= np.median (csv_vector,axis=0)
		med_lld= np.median (lld_vector,axis=0)
		iqr_lld= iqr(lld_vector,axis=0)
		med_prosody=np.median (prosody_vector,axis=0)
		med_vad= np.median(vad_vector,axis=0)
	else:
		med_csv=np.zeros(csv_vector.shape[1])
		med_lld=np.zeros(lld_vector.shape[1])
		iqr_lld=np.zeros(lld_vector.shape[1])
		med_prosody=np.zeros(prosody_vector.shape[1])
		med_vad=np.zeros(vad_vector.shape[1])
	
	res_vector= np.concatenate((med_csv,med_lld,iqr_lld,med_prosody,med_vad))
	return res_vector

def read_zface_features (zface_filename):

	res_vector=np.empty(0)

	mat=loadmat(zface_filename)
	zface_data = mat['fit']

	no_frames=zface_data.shape[1]
	isTracked_m  = zface_data[0]['isTracked']
	headPose_m   = zface_data[0]['headPose']
	#pts_3d_m= zface_data[0]['pts_3d']
	#pts_2d_m= zface_data[0]['pts_2d']
	pdmPars_m = zface_data[0]['pdmPars']
	no_pdm_parameters = 30

	isTracked = np.zeros(no_frames)
	#pts_3d= np.zeros((no_frames,512*3))
	#pts_2d= np.zeros((no_frames,49*2))
	headPose = np.zeros((no_frames,3) )
	pdmPars = np.zeros((no_frames,no_pdm_parameters) )
	track = np.zeros (no_frames)
	

	for ii in range (no_frames):
		isTracked[ii] = isTracked_m[ii][0]
		if isTracked[ii] != 0:
			track [ii] = 1 
			headPose[ii]  = headPose_m[ii].reshape(1,3)[0]
			pdmPars[ii]   = pdmPars_m[ii].reshape(1,no_pdm_parameters)[0]
			#pts_3d[ii]	  = pts_3d_m[ii].ravel()
			#pts_2d[ii]	  = pts_2d_m[ii].ravel()
	
	#print (zface_filename,no_frames,start_list,end_list)
	pdmPars = pdmPars[:,:15]
	
	if no_frames < 10000:
		return res_vector
	
	vector=np.concatenate((pdmPars,headPose),axis=1)   #Use this line to add as many zface_features as you want for thee raw zface vector
	
	return vector, track 

def read_AUs (AU_filename):

	res_vector=np.empty(0)

	mat=loadmat(AU_filename)
	au_data = mat['occurrence']

	no_frames=len(au_data)
	#print  (AU_filename,no_frames,start_list,end_list)

	if no_frames <10000:
		return res_vector
	
	vector = au_data.copy()

	return vector

def read_AU_intensity(AU_filename):

	res_vector=np.empty(0)
	mat= loadmat(AU_filename)
	#au_data= mat['intensity']

	au6= mat['AU6'][0][0][0][0].reshape(-1,1)
	au12=mat['AU12'][0][0][0][0].reshape(-1,1)

	au10=mat['AU10'][0][0][0][0].reshape(-1,1)
	au14=mat['AU14'][0][0][0][0].reshape(-1,1)

	#assert len(au6) == len(au12) == len(au10) == len(au14)

	no_frames=len(au6)
	if no_frames < 10000:
		return res_vector

	vector= np.concatenate((au6,au10,au12,au14),axis=1)

	return vector

def handle_landmarks (df):
	
	df, frame, tracked, keys= df[[x for x in df.keys() if x.startswith(' X') or x.startswith(' Y') or x.startswith(' Z')]].values, \
						df['frame'].values, \
						df[' success'].values, \
						[x for x in df.keys() if x.startswith(' X') or x.startswith(' Y') or x.startswith(' Z')]

	
	return df, frame,tracked, keys
def handle_AUs (df):
	df, frame, tracked,keys  = df [[x for x in df.keys() if x.startswith(' AU')]].values, \
						df['frame'].values, \
						df[' success'].values,\
						[x for x in df.keys() if x.startswith(' AU')]
	
	return df , frame, tracked, keys

def handle_gaze (df):
	df, frame, tracked, keys  = df [[x for x in df.keys() if x.startswith(' x') | x.startswith (' y') | x.startswith(' z')]].values,\
						 df['frame'].values, \
						 df[' success'].values, \
						 [x for x in df.keys() if x.startswith(' x') | x.startswith (' y') | x.startswith(' z')]
	
	return df, frame, tracked, keys
def extract_video (idx, val):
	print (idx, val)
	#if idx < 66:
	#	return
	
	x_df = pdf[pdf['ses_id']== val]

	c3_filename =  val  + '_CLNF_features3D.txt'
	c_filename = val + '_CLNF_features.txt'
	au_filename = val + '_CLNF_AUs.txt'
	gaze_filename = val + '_CLNF_gaze.txt'
	
	if sum(feature3D_files==c3_filename) != 1 :
		return
	if sum(feature_files == c_filename) != 1 :
		return
	if sum (au_files  == au_filename) != 1 :
		return
	if sum (gaze_files == gaze_filename) !=1 :
		return	
	c3_pdf = pd.read_csv  (os.path.join(feature3D_path, c3_filename))
	#c_pdf = pd.read_csv (os.path.join (feature_path, c_filename))
	au_pdf = pd.read_csv (os.path.join(au_path, au_filename))
	gaze_pdf = pd.read_csv (os.path.join (gaze_path, gaze_filename))
	
	c3_mat, c3_frame, c3_tracked, land_names = handle_landmarks (c3_pdf)
	au_mat, au_frame, au_tracked, au_names = handle_AUs (au_pdf)
	gaze_mat, gaze_frame, gaze_tracked , gaze_names= handle_gaze (gaze_pdf)

	start, end = x_df['start_time'], x_df['stop_time']
	start_frame, end_frame = (start*30).astype('int') +1, (end*30).astype('int')+1 

	video_feature_stack=[]
	for c_idx, (st, en) in enumerate(zip(start_frame, end_frame)):
		
		c3_landmarks = compute_statistics(c3_mat [st:en][ c3_tracked[st:en]==1].astype('float')) if sum(c3_tracked[st:en]==1) > 0 else compute_statistics(np.zeros(c3_mat.shape[1]))

		c3_au = compute_statistics (au_mat [st:en][ au_tracked[st:en]==1].astype('float')) if sum(au_tracked[st:en]==1) > 0 else compute_statistics(np.zeros(au_mat.shape[1]))
		c3_gaze = compute_statistics (gaze_mat[st:en] [ gaze_tracked[st:en]==1].astype('float')) if sum(gaze_tracked[st:en]==1) > 0 else compute_statistics(np.zeros(gaze_mat.shape[1]))
		video_vec = np.hstack ([c3_landmarks, c3_au, c3_gaze])
		video_feature_stack.append (video_vec)
	
	video_feature_stack = np.stack (video_feature_stack )

	land_names = [x+'_max' for x in land_names] + [x+'_mean' for x in land_names] + [x+'_std' for x in land_names] + [x+'_iqr' for x in land_names] + \
			[x+'_median' for x in land_names]
	au_names = [x+'_max' for x in au_names] + [x+'_mean' for x in au_names] + [x+'_std' for x in au_names] + [x+'_iqr' for x in au_names] + \
			[x+'_median' for x in au_names]
	gaze_names =[x+'_max' for x in gaze_names] + [x+'_mean' for x in gaze_names] + [x+'_std' for x in gaze_names] + [x+'_iqr' for x in gaze_names] + \
			[x+'_median' for x in gaze_names]
	names = land_names  + au_names + gaze_names

	names = ['openface_'+x for x in names]
	
	return video_feature_stack, names 

def compute_many_statistics(vector):
	
	if len(vector.shape) == 2  and vector.size >0 : # If it is a 2D array
		max_vector=np.max(vector,axis=0)
		mean_vector=np.mean(vector,axis=0)
		std_vector=np.std(vector,axis=0)
		iqr_vector=iqr (vector,axis=0)
		median_vector= np.median(vector,axis=0)
	elif len(vector.shape) == 1: # If only one frame comes so it comes as 1-D array
		max_vector=vector.copy()
		mean_vector= vector.copy()
		std_vector= np.zeros_like(max_vector)
		iqr_vector= np.zeros_like(max_vector)
		median_vector= vector.copy()
	else:
		max_vector=np.zeros (vector.shape[1])
		mean_vector=np.zeros_like(max_vector)
		std_vector=np.zeros_like(max_vector)
		iqr_vector=np.zeros_like(max_vector)
		median_vector=np.zeros_like(max_vector)

	stats_vec= np.hstack([max_vector,mean_vector,std_vector,iqr_vector,median_vector])
	return stats_vec

def compute_statistics(vector):

	if len(vector.shape) == 2  and vector.size >0 : # If it is a 2D array
		median_vector= np.median(vector,axis=0)
		std_vector = np.std (vector, axis=0)
	elif len(vector.shape) == 1: # If only one frame comes so it comes as 1-D array
		median_vector= vector.copy()
		std_vector = np.zeros_like (median_vector)
	else:
		median_vector=np.zeros (vector.shape[1])
		std_vector = np.zeros (vector.shape[1])

	res_vec = np.hstack ([median_vector, std_vector])
	return res_vec


def extract_features (s_id, file_list):
	window= 0.10

	total_family_id= []
	total_audio = []
	total_video = []
	total_text  = []
	total_speaker =[]
	total_gap =[]
	total_strategy= []

	for ii, val in enumerate (file_list):
		print (s_id, ii, val)
		
		x_df = pdf[pdf['family_id']== val]

		p_op_filename =  'TPOT_'+val+'_2'+'_2_all.csv'
		c_op_filename =  'TPOT_'+val+'_1'+'_2_all.csv'

		c_zface_filename =  val + '1'+ '_02_01_fit.mat'
		p_zface_filename =  val + '2'+ '_02_01_fit.mat'
		
		c_au_occ_filename = val + '1'+ '_02_01_au_out.mat'
		p_au_occ_filename = val + '2'+ '_02_01_au_out.mat'

		c_au_int_filename = val+ '1'+ '_02_01.mat'
		p_au_int_filename = val+ '2'+ '_02_01.mat'
	
	
		p_data = pd.read_csv  (os.path.join(audio_path, p_op_filename))
		c_data = pd.read_csv  (os.path.join(audio_path, c_op_filename))
	
		p_data = p_data.set_index ('Unnamed: 0')
		c_data = c_data.set_index ('Unnamed: 0')	
		p_data.index.names = [None]
		c_data.index.names = [None]
	
		[p_zface_mat, p_zface_track],[c_zface_mat, c_zface_track] = read_zface_features(os.path.join(feature_path,p_zface_filename)),\
																 read_zface_features(os.path.join(feature_path,c_zface_filename))

		p_au_occ_mat, 	c_au_occ_mat= read_AUs(os.path.join(au_occ_path,p_au_occ_filename)),\
									 read_AUs(os.path.join (au_occ_path,c_au_occ_filename))

		p_au_int_mat , c_au_int_mat=read_AU_intensity(os.path.join (au_int_path,p_au_int_filename)),\
									 read_AU_intensity(os.path.join (au_int_path,c_au_int_filename))		


		p_opensmile_lld_df= p_data[ [col for col in p_data if col.startswith('opensmile_eGeMAPSv01a_lld')] ] #111
		p_opensmile_csv_df= p_data[ [col for col in p_data if col.startswith('opensmile_eGeMAPSv01a') and not col.startswith('opensmile_eGeMAPSv01a_lld')]]
		p_prosody_df = p_data [ [col for col in p_data if col.startswith('opensmile_prosody')]]
		p_vad_df    = p_data  [ [col for col in p_data if col.startswith('opensmile_vad')]]
		#p_volume_df  = p_data[  [col for col in p_data if col.startswith('opensmile_volume')]]

		c_opensmile_lld_df= c_data[ [col for col in c_data if col.startswith('opensmile_eGeMAPSv01a_lld')] ] #111
		c_opensmile_csv_df= c_data[ [col for col in c_data if col.startswith('opensmile_eGeMAPSv01a') and not col.startswith('opensmile_eGeMAPSv01a_lld')]]
		c_prosody_df = c_data [ [col for col in c_data if col.startswith('opensmile_prosody')]]
		c_vad_df    = c_data  [ [col for col in c_data if col.startswith('opensmile_vad')]]
		#c_volume_df  = c_data[  [col for col in c_data if col.startswith('opensmile_volume')]]
		#volume_df  = p_data[  [col for col in p_data if col.startswith('opensmile_volume')]]
		
		
		start_utt, end_utt ,sub= x_df['onset'], x_df['offset'], x_df['speaker']
		start_frame, end_frame = (start_utt*30).astype('int'), (end_utt*30).astype('int')

		video_feature_stack=[]
		audio_feature_stack = []
	
		for jj, (start_idx, end_idx, st, end, sp) in enumerate (zip(start_utt, end_utt, start_frame, end_frame,sub)):
			#pdb.set_trace()
			
			segment_turn_start=[]
			segment_turn_end = []

			segment_turn_start_frame =[]
			segment_turn_end_frame =[]


			count_start = start_idx
			while count_start  < end_idx:
				if count_start + window < end_idx:
					segment_turn_start.append(count_start)
					segment_turn_end.append(count_start+window)
				else:
					segment_turn_start.append(count_start)
					segment_turn_end.append(end_idx)

				count_start += window

	
			count_start = st
			while count_start < end:
				if count_start +  int(window* 30) < end: 
					segment_turn_start_frame.append (count_start)
					segment_turn_end_frame.append(count_start+ int(window *  30 ))
				else:
					segment_turn_start_frame.append(count_start)
					segment_turn_end_frame.append(end)
				count_start = count_start + int(window*30)
			
			
			audio_vec = []
			for kk, (st_, en_)  in enumerate (zip (segment_turn_start, segment_turn_end)):
				if sp == 1:
					opensmile_vec= handle_opensmile_df([c_opensmile_lld_df, c_opensmile_csv_df, c_prosody_df, c_vad_df],st_,en_,skip=1)
				else:
					opensmile_vec= handle_opensmile_df([p_opensmile_lld_df, p_opensmile_csv_df, p_prosody_df, p_vad_df],st_,en_,skip=1)
				audio_vec.append (opensmile_vec)

			video_vec =[]
			for kk, (stf_, endf_) in enumerate(zip(segment_turn_start_frame, segment_turn_end_frame)):	
				if sp ==1 :	
					c3_landmarks = compute_statistics(c_zface_mat [stf_:endf_][ c_zface_track[stf_:endf_]==1].astype('float')) if sum(c_zface_track[stf_:endf_]==1) > 0 else compute_statistics(np.zeros(c_zface_mat.shape[1]))
					c3_au = compute_statistics (c_au_occ_mat [stf_:endf_][ c_zface_track[stf_:endf_]==1].astype('float')) if sum(c_zface_track[stf_:endf_]==1) > 0 else compute_statistics(np.zeros(c_au_occ_mat.shape[1]))
					c3_au_int = compute_statistics (c_au_int_mat [stf_:endf_][ c_zface_track[stf_:endf_]==1].astype('float')) if sum(c_zface_track[stf_:endf_]==1) > 0 else compute_statistics(np.zeros(c_au_int_mat.shape[1]))
					#c3_gaze = compute_statistics (gaze_mat[stf_:endf_] [ gaze_tracked[stf_:endf_]==1].astype('float')) if sum(gaze_tracked[stf_:endf_]==1) > 0 else compute_statistics(np.zeros(gaze_mat.shape[1]))
				
				else:
					c3_landmarks = compute_statistics(p_zface_mat [stf_:endf_][ p_zface_track[stf_:endf_]==1].astype('float')) if sum(p_zface_track[stf_:endf_]==1) > 0 else compute_statistics(np.zeros(p_zface_mat.shape[1]))
					c3_au = compute_statistics (p_au_occ_mat [stf_:endf_][ p_zface_track[stf_:endf_]==1].astype('float')) if sum(p_zface_track[stf_:endf_]==1) > 0 else compute_statistics(np.zeros(p_au_occ_mat.shape[1]))
					c3_au_int = compute_statistics (p_au_int_mat [stf_:endf_][ p_zface_track[stf_:endf_]==1].astype('float')) if sum(p_zface_track[stf_:endf_]==1) > 0 else compute_statistics(np.zeros(p_au_int_mat.shape[1]))
				
				feat_vec = np.hstack ([c3_landmarks, c3_au, c3_au_int])	
		
			
				video_vec.append (feat_vec)

			audio_vec = np.stack (audio_vec)
			video_vec = np.stack (video_vec)
			#print (audio_vec.shape, video_vec.shape)

			audio_feature_stack.append (audio_vec)
			video_feature_stack.append (video_vec)
		
		
		total_audio.append(audio_feature_stack)
		total_video.append(video_feature_stack)
		total_text.append (x_df['text'].values)
		total_family_id.append (val)
		total_speaker.append (x_df['speaker'].values)
		#total_gap.append (x_df['gap'].values)
		total_strategy.append (x_df['strategy'].values)
		#total_family_id.append (val)

	res= {}
	res['audio'] = total_audio
	res['video'] = total_video
	res['text']  = total_text
	res['family_id'] = total_family_id
	res['speaker'] = total_speaker
	res['strategy'] = total_strategy
	
	return res 
def get_all_families (data, splits=10):
	
	all_data =[]
	all_data = np.array(data)
	idx = int (len(all_data)/splits)
	
	splits = []
	for i in range(0, len(all_data), idx):
		
		splits.append( all_data [i:i+idx])


	return splits

data_path = '../study/'
pdf = pd.read_csv (os.path.join (data_path, 'data.csv'), dtype={'family_id':object})
#pdf = pdf.dropna()


splits = get_all_families(np.unique(pdf['family_id']))

def par_fold(idx,split):

	print ('Fold ', idx)


	res = extract_features  (idx,split)
	print (len(res['family_id']))
	np.save(output_path+'split_' + str(idx) +'.npy', res)


d = Parallel(n_jobs=12)(delayed(par_fold)(idx,files) for idx, files in enumerate(splits))

#res = make ()
print ("Done ")