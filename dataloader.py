import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd

#from transformers import BertTokenizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA 

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pdb 

import numpy as np 
import itertools
import pymrmr
import random 
#from syntactic_features import *


opensmile_df =['GeMAPS_lld_Loudness_sma3', 'GeMAPS_lld_alphaRatio_sma3',
       'GeMAPS_lld_hammarbergIndex_sma3', 'GeMAPS_lld_slope0-500_sma3',
       'GeMAPS_lld_slope500-1500_sma3', 'GeMAPS_lld_spectralFlux_sma3',
       'GeMAPS_lld_mfcc1_sma3', 'GeMAPS_lld_mfcc2_sma3',
       'GeMAPS_lld_mfcc3_sma3', 'GeMAPS_lld_mfcc4_sma3',
       'GeMAPS_lld_F0semitoneFrom27.5Hz_sma3nz',
       'GeMAPS_lld_jitterLocal_sma3nz', 'GeMAPS_lld_shimmerLocaldB_sma3nz',
       'GeMAPS_lld_HNRdBACF_sma3nz', 'GeMAPS_lld_logRelF0-H1-H2_sma3nz',
       'GeMAPS_lld_logRelF0-H1-A3_sma3nz', 'GeMAPS_lld_F1frequency_sma3nz',
       'GeMAPS_lld_F1bandwidth_sma3nz',
       'GeMAPS_lld_F1amplitudeLogRelF0_sma3nz',
       'GeMAPS_lld_F2frequency_sma3nz',
       'GeMAPS_lld_F2amplitudeLogRelF0_sma3nz',
       'GeMAPS_lld_F3frequency_sma3nz',
       'GeMAPS_lld_F3amplitudeLogRelF0_sma3nz'
    ]
opensmile_csv_df =['GeMAPS_F0semitoneFrom27.5Hz_sma3nz_amean',
       'GeMAPS_F0semitoneFrom27.5Hz_sma3nz_stddevNorm',
       'GeMAPS_F0semitoneFrom27.5Hz_sma3nz_percentile20.0',
       'GeMAPS_F0semitoneFrom27.5Hz_sma3nz_percentile50.0',
       'GeMAPS_F0semitoneFrom27.5Hz_sma3nz_percentile80.0',
       'GeMAPS_F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2',
       'GeMAPS_F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope',
       'GeMAPS_F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope',
       'GeMAPS_F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope',
       'GeMAPS_F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope',
       'GeMAPS_loudness_sma3_amean', 'GeMAPS_loudness_sma3_stddevNorm',
       'GeMAPS_loudness_sma3_percentile20.0',
       'GeMAPS_loudness_sma3_percentile50.0',
       'GeMAPS_loudness_sma3_percentile80.0',
       'GeMAPS_loudness_sma3_pctlrange0-2',
       'GeMAPS_loudness_sma3_meanRisingSlope',
       'GeMAPS_loudness_sma3_stddevRisingSlope',
       'GeMAPS_loudness_sma3_meanFallingSlope',
       'GeMAPS_loudness_sma3_stddevFallingSlope',
       'GeMAPS_spectralFlux_sma3_amean', 'GeMAPS_spectralFlux_sma3_stddevNorm',
       'GeMAPS_mfcc1_sma3_amean', 'GeMAPS_mfcc1_sma3_stddevNorm',
       'GeMAPS_mfcc2_sma3_amean', 'GeMAPS_mfcc2_sma3_stddevNorm',
       'GeMAPS_mfcc3_sma3_amean', 'GeMAPS_mfcc3_sma3_stddevNorm',
       'GeMAPS_mfcc4_sma3_amean', 'GeMAPS_mfcc4_sma3_stddevNorm',
       'GeMAPS_jitterLocal_sma3nz_amean',
       'GeMAPS_jitterLocal_sma3nz_stddevNorm',
       'GeMAPS_shimmerLocaldB_sma3nz_amean',
       'GeMAPS_shimmerLocaldB_sma3nz_stddevNorm',
       'GeMAPS_HNRdBACF_sma3nz_amean', 'GeMAPS_HNRdBACF_sma3nz_stddevNorm',
       'GeMAPS_logRelF0-H1-H2_sma3nz_amean',
       'GeMAPS_logRelF0-H1-H2_sma3nz_stddevNorm',
       'GeMAPS_logRelF0-H1-A3_sma3nz_amean',
       'GeMAPS_logRelF0-H1-A3_sma3nz_stddevNorm',
       'GeMAPS_F1frequency_sma3nz_amean',
       'GeMAPS_F1frequency_sma3nz_stddevNorm',
       'GeMAPS_F1bandwidth_sma3nz_amean',
       'GeMAPS_F1bandwidth_sma3nz_stddevNorm',
       'GeMAPS_F1amplitudeLogRelF0_sma3nz_amean',
       'GeMAPS_F1amplitudeLogRelF0_sma3nz_stddevNorm',
       'GeMAPS_F2frequency_sma3nz_amean',
       'GeMAPS_F2frequency_sma3nz_stddevNorm',
       'GeMAPS_F2bandwidth_sma3nz_amean',
       'GeMAPS_F2bandwidth_sma3nz_stddevNorm',
       'GeMAPS_F2amplitudeLogRelF0_sma3nz_amean',
       'GeMAPS_F2amplitudeLogRelF0_sma3nz_stddevNorm',
       'GeMAPS_F3frequency_sma3nz_amean',
       'GeMAPS_F3frequency_sma3nz_stddevNorm',
       'GeMAPS_F3bandwidth_sma3nz_amean',
       'GeMAPS_F3bandwidth_sma3nz_stddevNorm',
       'GeMAPS_F3amplitudeLogRelF0_sma3nz_amean',
       'GeMAPS_F3amplitudeLogRelF0_sma3nz_stddevNorm',
       'GeMAPS_alphaRatioV_sma3nz_amean',
       'GeMAPS_alphaRatioV_sma3nz_stddevNorm',
       'GeMAPS_hammarbergIndexV_sma3nz_amean',
       'GeMAPS_hammarbergIndexV_sma3nz_stddevNorm',
       'GeMAPS_slopeV0-500_sma3nz_amean',
       'GeMAPS_slopeV0-500_sma3nz_stddevNorm',
       'GeMAPS_slopeV500-1500_sma3nz_amean',
       'GeMAPS_slopeV500-1500_sma3nz_stddevNorm',
       'GeMAPS_spectralFluxV_sma3nz_amean',
       'GeMAPS_spectralFluxV_sma3nz_stddevNorm', 'GeMAPS_mfcc1V_sma3nz_amean',
       'GeMAPS_mfcc1V_sma3nz_stddevNorm', 'GeMAPS_mfcc2V_sma3nz_amean',
       'GeMAPS_mfcc2V_sma3nz_stddevNorm', 'GeMAPS_mfcc3V_sma3nz_amean',
       'GeMAPS_mfcc3V_sma3nz_stddevNorm', 'GeMAPS_mfcc4V_sma3nz_amean',
       'GeMAPS_mfcc4V_sma3nz_stddevNorm', 'GeMAPS_alphaRatioUV_sma3nz_amean',
       'GeMAPS_hammarbergIndexUV_sma3nz_amean',
       'GeMAPS_slopeUV0-500_sma3nz_amean',
       'GeMAPS_slopeUV500-1500_sma3nz_amean',
       'GeMAPS_spectralFluxUV_sma3nz_amean', 'GeMAPS_loudnessPeaksPerSec',
       'GeMAPS_VoicedSegmentsPerSec', 'GeMAPS_MeanVoicedSegmentLengthSec',
       'GeMAPS_StddevVoicedSegmentLengthSec',
       'GeMAPS_MeanUnvoicedSegmentLength',
       'GeMAPS_StddevUnvoicedSegmentLength',
       'GeMAPS_equivalentSoundLevel_dBp']


prosody_df =['prosodyAcf_voiceProb_sma', 'prosodyAcf_F0_sma',
       'prosodyAcf_pcm_loudness_sma']
vad_df =['vad_df']


covarep_df = ['covarep_vowelSpace',
            'covarep_MCEP_0',
            'covarep_MCEP_1',
            'covarep_VAD',
            'covarep_f0',
            'covarep_NAQ',
            'covarep_QOQ',
            'covarep_MDQ',
            'covarep_peakSlope',
            'covarep_F1',
            'covarep_F2']
covarep_names= ('Median of '+pd.Series(covarep_df) ).tolist() + ('IQR of' +pd.Series(covarep_df[4:])).tolist()
opensmile_names =('Median of '+pd.Series(opensmile_csv_df)).tolist() + ('Median of' + pd.Series(opensmile_df)). tolist() + ('IQR of'+ pd.Series(opensmile_df)).tolist() \
                + ('Median of '+ pd.Series(prosody_df)).tolist() + ('Median Of '+pd.Series(vad_df)).tolist() 


audio_names = covarep_names + opensmile_names

#-----------Definition of Video names -------------------------#
zface_feat = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','head_pitch','head_yaw','head_roll']
au_occ_feat = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU23', 'AU24']
au_int_feat = ['int_6', 'int_10', 'int_12', 'int_14']

video_names= ('Max of ' + pd.Series(zface_feat)).tolist() \
            + ('Mean of ' + pd.Series(zface_feat)).tolist() \
            + ('Std of '+pd.Series(zface_feat)).tolist()\
            +  ('IQR of '+ pd.Series(zface_feat)).tolist() \
            + ('Median of ' + pd.Series(zface_feat)).tolist() \
            + ('Max of ' + pd.Series(au_occ_feat)).tolist() \
            + ('Mean of ' + pd.Series(au_occ_feat)).tolist()  \
            + ('Std of '+pd.Series(au_occ_feat)).tolist()  \
            + ('IQR of '+ pd.Series(au_occ_feat)).tolist() \
            + ('Median of ' + pd.Series(au_occ_feat)).tolist() \
            + ('Max of ' + pd.Series(au_int_feat)).tolist() \
            + ('Mean of' + pd.Series(au_int_feat)).tolist()  \
            + ('Std of'+pd.Series(au_int_feat)).tolist()  \
            + ('IQR of'+ pd.Series(au_int_feat)).tolist()\
            + ('Median of' + pd.Series(au_int_feat)).tolist()


#---Addition 
#video_names = video_names + ['IQR of gaze_x', 'IQR for gaze_y']
# Load the BERT tokenizer
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # Create a function to tokenize a set of texts
def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs

    
        encoded_sent = tokenizer.encode_plus(
            text=sent,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=128,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            truncation='longest_first',
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )
    
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors


    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

def text_preprocessor_idf (train, val, test):
        #without smooth IDF
    print("Without Smoothing:")
    #define tf-idf
    '''
     tf_idf_vec = TfidfVectorizer(use_idf=True, 
                            smooth_idf=False,  
                            ngram_range=(1,1),stop_words='english') # to use only  bigrams ngram_range=(2,2)
    #transform
    tf_idf_data = tf_idf_vec.fit_transform(text)
    
    #create dataframe
    tf_idf_dataframe=pd.DataFrame(tf_idf_data.toarray(),columns=tf_idf_vec.get_feature_names())
    print(tf_idf_dataframe)
    print("\n")
     
    #with smooth
    '''
    tf_idf_vec_smooth = TfidfVectorizer(use_idf=True,  
                            smooth_idf=True,  
                            ngram_range=(1,1),stop_words='english')
     
     
    tf_idf_data_smooth = tf_idf_vec_smooth.fit(train)

    train_dat = tf_idf_data_smooth.transform(train).toarray()
    val_dat = tf_idf_data_smooth.transform(val).toarray()
    test_dat = tf_idf_data_smooth.transform(test).toarray()

  
    pca = PCA(n_components = 200, svd_solver='full')
    pca_fit= pca.fit(train_dat)
    
    train_dat = pca_fit.transform(train_dat)
    val_dat = pca_fit.transform(val_dat)
    test_dat = pca_fit.transform(test_dat)
  
    return train_dat, val_dat, test_dat

def text_preprocessor (text):
    CountVec = CountVectorizer(ngram_range=(1,1), # to use bigrams ngram_range=(2,2)
                       stop_words='english')
    count_data = CountVec.fit_transform(text)

    pdb.set_trace()
 

    return 
def analyze (feature, label, top_k=50):

    feature_set = label.reshape(-1,1)
    '''
    for idx,subset in enumerate(itertools.combinations (audio_names+video_names, top_k)):
      print (idx)
    '''
    
    if len (feature) > 1:
        for f in feature:
            feature_set = np.hstack ([feature_set, f]) if feature_set is not None else f 

    feature_name = ['label'] + audio_names + video_names
    
    df = pd.DataFrame( data = feature_set, index=None, columns= feature_name)

    x= pymrmr.mRMR(df, 'MIQ', top_k)
    pdb.set_trace()

    return 
def prune_features (audio, video, feature_name):

  #global_feature_names = audio_names + video_names
  
  a_n= np.array(audio_names)
  v_n = np.array(video_names)

  def read_feature_file(filename):
    df = pd.read_csv(filename, '\t', header=None, names=['Names', 'Score'])

    names = df['Names'].values.tolist()
    return names 

  names = read_feature_file(feature_name)
  names = names [:50]
  new_audio = []
  new_video = []
  
  for ele in names:

    if ele in a_n:
      try:  
        new_audio.append(audio [:, np.where(a_n == ele)[0][0]])
      except:
        pdb.set_trace()
    elif ele in v_n:
      try:
        new_video.append(video [:, np.where(v_n == ele)[0][0]])
      except:
        pdb.set_trace()
  new_audio= np.array(new_audio).T
  new_video= np.array(new_video).T

  return new_audio, new_video 


class TPOT_loader (Dataset):
  def __init__(self, full_data, select_indices,  domain, train=True, scaler=None, dialog=True):

      data = np.load (full_data,allow_pickle=True).item()

      audio = data['audio']
      video = data['video']
      text  = data['sequence']
      speaker = data['speaker']
      strategy = data['strategy']
      ses_id = data['family_id']
    
      if dialog:
        self.audio = np.concatenate (audio)
        self.video = np.concatenate (video)
        self.text  = np.concatenate (text)
        self.speaker = np.concatenate (speaker)
        self.strategy = np.concatenate (strategy)
        self.ses = np.concatenate([[x] * len(audio[i]) for i,x in enumerate(ses_id)])

      #self.text, self.mask = preprocessing_for_bert (self.text_unravel)

      indices= self.strategy != 2

      self.audio = self.audio [indices]
      self.video = self.video [indices]
      self.text= self.text[indices]
      self.speaker = self.speaker [indices]
      self.strategy = self.strategy [indices]
      self.ses = self.ses[indices]
      
      self.len = len(self.ses)
     
  def __len__(self):
    return self.len 

  def __getitem__(self, index):
      
      return torch.FloatTensor (self.audio[index]),\
             torch.FloatTensor (self.video[index]),\
             torch.LongTensor (self.text[index]),\
             self.strategy[index], \
             self.ses[index]
  def collate_fn(self, data):
      dat = pd.DataFrame(data)
      
      # Lengths of the individual sequences ----#

      #text = [ torch.LongTensor(x) for x in dat[3]]
      label = torch.LongTensor ([d for d in dat[3]]) #4 is srategy

      a_mask = [ len(x) for x in dat[0]]  
      v_mask = [ len(x) for x in dat[1]]
      t_mask = [ len(x) for x in dat[2]]

      
      return pad_sequence(dat[0],True), pad_sequence(dat[1],True),\
           pad_sequence (dat[2],True), label,torch.LongTensor(a_mask), torch.LongTensor(v_mask), \
           torch.LongTensor(t_mask), dat[4].tolist()

class TPOTDataset(Dataset):

    def __init__(self, full_data, select_indices,  domain, train=True, scaler=None, dialog=False):

        data = np.load(full_data, allow_pickle=True).item()

        #self.audio_features= np.load(audio_path, allow_pickle=True).item()
        #self.video_features= np.load(video_path, allow_pickle=True).item()
        #self.text_features = np.load (text_path, allow_pickle=True).item()

        audio = data['audio']
        video = data['video']
        l_audio= data['listener_audio']
        l_video= data['listener_video']
        text = data['turn_text']
        text_seq  = data['turn_sequence']
        #label = data['turn_label'] 
        strategy = data['turn_strategy']
        speaker = data['turn_speaker']
        family = data['turn_filename']
        duration = data['turn_duration']   
        #gap = data['speaker_gap']  
          
        #self.tokenizer = tokenizer
        #self.max_len= 160        
        
        #----------Now we have to unpack from dialog level to turn level ------------#
        self.audio= np.concatenate(audio)
        self.video= np.concatenate(video)
        self.text = np.concatenate(text)
        self.text_seq = np.concatenate(text_seq)
        #self.label = np.concatenate(label)
        self.strategy = np.concatenate(strategy)
        self.speaker = np.concatenate(speaker)
        self.family = np.concatenate([ [family[i]] * len(x) for i,x in enumerate(audio) ])
        self.duration= np.concatenate(duration)
        self.l_video = np.concatenate(l_video)
        self.l_audio = np.concatenate(l_audio)
 
   

        self.audio, self.video = prune_features(self.audio, self.video, 'feat.txt')
        self.l_audio, self.l_video = prune_features (self.l_audio, self.l_video, 'feat.txt') 
   
        self.len = len(self.family)
        "Scaling the data"
        if train:
          self.scaler = self._audiovideo_(scaler=None)
        else:
          self._audiovideo_(scaler=scaler) 


      
    def _audiovideo_(self, scaler=None):
      if scaler is None :
        audio_scaler = StandardScaler()
        video_scaler = StandardScaler()
        

        fit_audio = audio_scaler.fit(self.audio)
        fit_video = video_scaler.fit(self.video)
        
        self.audio= fit_audio.transform(self.audio)
        self.video= fit_video.transform(self.video)
        return [fit_audio, fit_video]

      else:
        [fit_audio,fit_video] = scaler
        
        self.audio= fit_audio.transform(self.audio)
        self.video= fit_video.transform(self.video)
 
    def __getitem__(self, index):
        
        return torch.FloatTensor (self.audio[index]),\
               torch.FloatTensor (self.video[index]),\
               torch.FloatTensor (self.l_audio[index]), \
               torch.FloatTensor (self.l_video[index]), \
               self.strategy[index], \
               self.family[index]

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        
        # Lengths of the individual sequences ----#
 
        #text = [ torch.LongTensor(x) for x in dat[3]]
        label = torch.LongTensor ([d for d in dat[4]]) #4 is srategy

      
        
        return pad_sequence(dat[0],True), pad_sequence(dat[1],True),\
             pad_sequence (dat[2],True), pad_sequence(dat[3],True), label, dat[5].tolist()

class TPOTDataset_MLP(Dataset):

    def __init__(self, full_data, select_indices,  domain, train=True, scaler=None, dialog=False):
        [data_dir, feat_dir] = full_data
        data = np.load(data_dir, allow_pickle=True).item()

        feat = np.load (feat_dir, allow_pickle=True).item()

        #self.audio_features= np.load(audio_path, allow_pickle=True).item()
        #self.video_features= np.load(video_path, allow_pickle=True).item()
        #self.text_features = np.load (text_path, allow_pickle=True).item()
      
        speaker_track = feat['speaker']
        listener_track = feat['listener']
        
        text = data['turn_text']
        text_seq  = data['turn_sequence']
        #label = data['turn_label'] 
        strategy = data['turn_strategy']
        speaker = data['turn_speaker']
        family = data['turn_filename']
        duration = data['turn_duration']   
        #gap = data['speaker_gap']  
          
        #self.tokenizer = tokenizer
        #self.max_len= 160        
        
        #----------Now we have to unpack from dialog level to turn level ------------#
        self.feat = np.concatenate(speaker_track)
        self.l_feat = np.concatenate(listener_track)
      
        self.text = np.concatenate(text)
        self.text_seq = np.concatenate(text_seq)
        #self.label = np.concatenate(label)
        self.strategy = np.concatenate(strategy)
        self.speaker = np.concatenate(speaker)
        self.family = np.concatenate([ [family[i]] * len(x) for i,x in enumerate(speaker_track) ])
        self.duration= np.concatenate(duration)


        #---------------Selection 
        indices = self.strategy != 2
        self.feat= self.feat[indices]
        self.l_feat = self.l_feat [indices]
        self.text = self.text [indices]
        self.text_seq= self.text_seq [indices]
        self.strategy = self.strategy[indices]
        self.family = self.family[indices]
        self.duration = self.duration[indices]

      
        text_df = pd.DataFrame (self.text, columns =['feature'])
        text_feat = lexical_features (text_df, 'feature', self.duration)
        self.text = text_feat
    
      
        #self.audio, self.video = prune_features(self.audio, self.video, 'feat.txt')
        #self.l_audio, self.l_video = prune_features (self.l_audio, self.l_video, 'feat.txt') 
   
        self.len = len(self.family)
        "Scaling the data"
        if train:
          self.scaler = self._audiovideo_(scaler=None)
        else:
          self._audiovideo_(scaler=scaler) 


      
    def _audiovideo_(self, scaler=None):
      if scaler is None :
        text_scaler = StandardScaler()
      
        fit_text = text_scaler.fit(self.text)
        
        self.text= fit_text.transform(self.text)
      
        return [fit_text]

      else:
        [fit_text] = scaler
        
        self.text= fit_text.transform(self.text)
 
    def __getitem__(self, index):
        
        return torch.FloatTensor (self.feat[index]),\
               torch.FloatTensor (self.l_feat[index]),\
               torch.FloatTensor (self.text[index]),\
               self.strategy[index], \
               self.family[index]

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        
        # Lengths of the individual sequences ----#
 
        #text = [ torch.LongTensor(x) for x in dat[3]]
        label = torch.LongTensor ([d for d in dat[3]]) #4 is srategy

      
      
        return pad_sequence(dat[0],True), pad_sequence(dat[1],True),\
             pad_sequence (dat[2],True), label, dat[4].tolist()
    


