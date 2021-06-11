

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.utils as utils
import numpy as np 
import pdb 

from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_
from torch.autograd import Variable






class Attention(nn.Module):
    
    def __init__(self, query_dim, key_dim, value_dim=None, att_type= 'dot' , device='cpu'):
        super().__init__()
        
        self.att_type = att_type
        if self.att_type == 'scaled_dot':
            self.query_transform = nn.Linear(query_dim, key_dim)
            self.scale = 1.0/np.sqrt(query_dim)
       
        self.softmax = nn.Softmax(dim=2)
        self.device = device
        

    def forward(self, mask, query, keys, values=None, input_lens=None):
        # query: [B,Q] (hidden state, decoder output, etc.)
        # keys: [T,B,K] (encoder outputs which are hiddden states of LSTM ,etc.)
        # values: [T,B,V] (encoder outputs)  ==== We don't need values.     
        # assume Q == K
    
        # compute energy = = attention weights
        if self.att_type == 'dot':
            query = self.query_transform(query).unsqueeze(1) # [B,Q] ->  [B,K] -- >  [B,1,K]    
            keys = keys.permute(0,2,1) # [B,T,K] -> [B,K,T]
            energy = torch.bmm(query, keys) # [B,1,K]*[B,K,T] = [B,1,T]
            mask = torch.arange(keys.size(2)).unsqueeze(0).to(self.device) < input_lens.unsqueeze(1)  #  1,T  <  B,1  = B, 1, T          
            mask = mask.unsqueeze(1)
            energy[~mask]= float('-inf')
            energy = self.softmax(energy)
            context_vec = torch.bmm(energy, keys.permute(0,2,1)).squeeze(1) # [B,1,T]*[B,T,V] -> [B,V]
        elif self.att_type == 'scaled_dot':
           
            query = self.query_transform(query).unsqueeze(1) # [B,Q] ->  [B,K] -- >  [B,1,K]
          
            keys = keys.permute(0,2,1) # [B,T,K] -> [B,K,T]
            energy = torch.bmm(query, keys) # [B,1,K]*[B,K,T] = [B,1,T]
            energy=  energy.mul_(self.scale)
            if input_lens is not None : #Mask is required
                mask = torch.arange(keys.size(2)).unsqueeze(0).to(self.device) < input_lens.unsqueeze(1)  #  1,T  <  B,1  = B, 1, T          
                mask = mask.unsqueeze(1)
            energy[~mask]= float('-inf')
            energy = self.softmax(energy)
            context_vec = torch.bmm(energy, keys.permute(0,2,1)).squeeze(1) # [B,1,T]*[B,T,V] -> [B,V]
     
        return (context_vec, energy)

class SimpleModel (nn.Module):
    def __init__ (self, D_a, D_v, vocab_size, embedding_dim, gap_dim,  H_a, H_v, H_t , H_g, emo_dim, speaker_dim, class_hid, n_classes=2, att_type=None,modal=None, drop=0.3, device = 'cpu'):
        super(SimpleModel, self).__init__()
        self.audio_rnn = nn.LSTM(input_size=D_a, hidden_size=H_a, num_layers=2, dropout=0.3, bidirectional=False, batch_first=True)
        self.video_rnn = nn.LSTM(input_size=D_v, hidden_size=H_v, num_layers=2, dropout=0.3, bidirectional=False, batch_first=True)
        self.text_rnn = nn.LSTM(input_size= embedding_dim, hidden_size=H_t, num_layers=2, dropout=0.3, bidirectional=False, batch_first=True)
        self.gap_rnn = nn.LSTM(input_size= gap_dim, hidden_size= H_g, num_layers=2, dropout=0.3, bidirectional=False, batch_first=True)
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.device= device
        self.att_type= att_type   
        self.modal = modal      

        #self.fc1= nn.Linear (H_a+H_v+H_t+emo_dim+speaker_dim , class_hid)
        #self.fc1= nn.Linear (H_a+H_v+H_t , class_hid)
        #self.fc1= nn.Linear (H_a+H_v , class_hid)
    
        #self.fc1 = nn.Linear (H_t, class_hid)
        '''
        if self.modal == 'text':
            self.fc1= nn.Linear (H_t , class_hid)
        elif self.modal == 'speech_text':
            self.fc1= nn.Linear (H_a + H_t , class_hid)
        else:
            self.fc1= nn.Linear (H_a  +H_v  +H_t + H_g , class_hid)
        '''
        self.fc1= nn.Linear (H_a  +H_v  +H_t + H_g , class_hid)
        
        

        self.text_norm = nn.BatchNorm1d(H_t)
        self.audio_norm = nn.BatchNorm1d(H_a)
        self.video_norm = nn.BatchNorm1d(H_v)
        self.gap_norm = nn.BatchNorm1d(H_g)
        self.drop = nn.Dropout(drop)
        '''     
        self.fc2= nn.Linear (class_hid, 128)
        self.fc3= nn.Linear (128, n_classes)
        self.dropout= nn.Dropout(0.3)
        '''
        if att_type is not None:
            self.text_attention = Attention(query_dim=H_t, key_dim=H_t, value_dim=None , att_type='scaled_dot' ,device=self.device)
            self.audio_attention = Attention(query_dim= H_a, key_dim= H_a, value_dim= None, att_type='scaled_dot', device=self.device)
            self.video_atttention = Attention (query_dim=H_v, key_dim= H_v, value_dim= None, att_type='scaled_dot', device=self.device)
            self.gap_atttention = Attention (query_dim=H_g, key_dim= H_g, value_dim= None, att_type='scaled_dot', device=self.device)

    def init_pretrained_embeddings_from_numpy(self, pretrained_word_vectors):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        # if is_static:
        self.embedding.weight.requires_grad = False


    def forward (self, audio_info, video_info, text_info, gap_info, emo_feat=None, speaker_feat=None, umask=None):
        text_feat, text_lens   = text_info
        audio_feat, audio_lens = audio_info
        video_feat, video_lens = video_info
        gap_feat, gap_lens = gap_info
    
        
        text_feat= self.embedding (text_feat)
        try:    
            text_rnn_inp = utils.rnn.pack_padded_sequence(text_feat, lengths=text_lens, batch_first=True, enforce_sorted=False)
            text_out, (text_hid, text_cell) = self.text_rnn(text_rnn_inp)
            text_hid_states, _ = pad_packed_sequence(text_out, batch_first=True)   
            
            
            audio_rnn_inp = utils.rnn.pack_padded_sequence(audio_feat, lengths=audio_lens, batch_first=True, enforce_sorted=False)
            audio_out, (audio_hid, audio_cell) = self.audio_rnn(audio_rnn_inp)
            audio_hid_states,_ = pad_packed_sequence (audio_out, batch_first=True)

            video_rnn_inp = utils.rnn.pack_padded_sequence(video_feat, lengths=video_lens, batch_first=True, enforce_sorted=False)   
            video_out, (video_hid, video_cell) = self.video_rnn(video_rnn_inp)
            video_hid_states, _ = pad_packed_sequence (video_out, batch_first=True)
            
            gap_rnn_inp = utils.rnn.pack_padded_sequence(gap_feat, lengths=gap_lens, batch_first=True, enforce_sorted=False)   
            gap_out, (gap_hid, gap_cell) = self.gap_rnn(gap_rnn_inp)
            gap_hid_states, _ = pad_packed_sequence (gap_out, batch_first=True)
        
        except:
            pdb.set_trace()    
        if self.att_type is not None :
            text_context, _ = self.text_attention (None, query=text_hid [-1], keys= text_hid_states, values = None, input_lens= text_lens)
            audio_context, _ = self.audio_attention(None, query = audio_hid[-1], keys = audio_hid_states, values=None, input_lens = audio_lens)
            video_context, _ = self.video_atttention (None, query= video_hid[-1], keys = video_hid_states, values=None, input_lens= video_lens)
            video_context, _ = self.gap_atttention (None, query= gap_hid[-1], keys = gap_hid_states, values=None, input_lens= gap_lens)
            if self.modal == 'text':
                whole_feat = text_context 
            elif self.modal == 'speech_text':   
              whole_feat = torch.cat ( (text_context,audio_context ), dim=-1)
            else:
              whole_feat = torch.cat ( (text_context, audio_context, video_context, gap_context), dim=-1)
        else:
            if self.modal == 'text':
                whole_feat = text_hid[-1]
            elif self.modal == 'speech_text':
                whole_feat = torch.cat ((text_hid[-1], audio_hid[-1]), dim=-1)
            else:
              
                whole_feat = torch.cat ( (text_hid[-1],audio_hid[-1], video_hid[-1], gap_hid[-1]), dim=-1)
        
        #whole_feat = text_hid[-1]
        '''
        whole_feat = F.relu(self.dropout(self.fc1 (whole_feat)))
        whole_feat = F.relu(self.fc2(whole_feat))
        whole_feat = F.relu(self.fc3(whole_feat))
        
        x = F.log_softmax(whole_feat,1)
        '''
        text_norm = self.drop(self.text_norm (text_hid[-1]))
        audio_norm = self.drop(self.audio_norm (audio_hid[-1]))
        video_norm = self.drop(self.video_norm(video_hid[-1]))
        gap_norm = self.drop(self.gap_norm (gap_hid[-1]))


        return text_norm, audio_norm, video_norm, gap_norm


class SubNet(nn.Module):
    '''
    The subnetwork that is used in LMF for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3


class TextSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in LMF for text
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1


class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, submodel_paramms, input_dims=None, hidden_dims=None, text_out=None, dropouts=None, output_dim=None, rank=None, use_softmax=False, dropout=0.3):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, hidden dims of the sub-networks
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(LMF, self).__init__()


        [D_a, D_v, vocab_size, embedding_dim, gap_dim, H_a, H_v, H_t, H_g, emo_dim, speaker_dim, class_hid, n_classes, modal, device, glove_matrix]= submodel_paramms
        self.submodel = SimpleModel ( D_a, D_v, vocab_size, embedding_dim , gap_dim, H_a, H_v, H_t, H_g, emo_dim, speaker_dim,
                            class_hid= class_hid,
                            n_classes= n_classes,
                            #att_type = 'scaled_dot',
                            att_type = None,
                            modal =  modal,
                            drop = dropout, 
                            device = device)
        self.submodel.init_pretrained_embeddings_from_numpy(glove_matrix)

        self.device = device
        # dimensions are specified in the order of audio, video and text
        '''
        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]
        self.text_in = input_dims[2]

        self.audio_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]
        self.text_hidden = hidden_dims[2]


        self.audio_prob = dropouts[0]
        self.video_prob = dropouts[1]
        self.text_prob = dropouts[2]
    
    	self.text_out= text_out
        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)

        '''
 
        self.output_dim = n_classes
        self.rank = rank
        self.use_softmax = use_softmax

        self.post_fusion_prob = dropout

        self.audio_hidden = H_a
        self.video_hidden = H_v
        self.text_hidden = H_t

      
        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        # self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.audio_factor = Parameter(torch.Tensor(self.rank, self.audio_hidden + 1, self.output_dim))
        self.video_factor = Parameter(torch.Tensor(self.rank, self.video_hidden + 1, self.output_dim))
        self.text_factor = Parameter(torch.Tensor(self.rank, self.text_hidden + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))
        
        # init teh factors
        xavier_normal_(self.audio_factor)
        xavier_normal_(self.video_factor)
        xavier_normal_(self.text_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self,  audio_info, video_info, text_info, gap_info, emo_feat=None, speaker_feat=None, umask=None):

        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        text_h , audio_h, video_h, gap_h = self.submodel (audio_info, video_info, text_info, gap_info)
       
        #audio_h = self.audio_subnet(audio_x)
        #video_h = self.video_subnet(video_x)
        #text_h = self.text_subnet(text_x)
        batch_size = audio_h.data.shape[0]
        
        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE).to(self.device), requires_grad=False), audio_h), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE).to(self.device), requires_grad=False), video_h), dim=1)
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE).to(self.device), requires_grad=False), text_h), dim=1)
        
        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_text = torch.matmul(_text_h, self.text_factor)
        
        fusion_zy = fusion_audio #* fusion_video * fusion_text
        
        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        
        output = output.view(-1, self.output_dim)
       	
        if self.use_softmax:
            output = F.log_softmax(output, dim=1)
        
        return output


class MaskedLoss (nn.Module):

    def __init__(self, weight=None):
        super(MaskedLoss, self).__init__()
        self.weight=weight

        self.loss  = nn.NLLLoss(weight=self.weight, reduction='sum')


    def forward(self,pred, target, mask=None):
        
        if mask is not None: 
            mask_ = mask.view(-1,1)
           
            if type(self.weight) == type(None):
                if torch.sum(mask_) != 0:
                    loss = self.loss (pred * mask_,target) / torch.sum(mask_)
                else:
                    loss = self.loss (pred * mask_,target) / 0.001

            else:
                
                loss = self.loss (pred*mask_, target) \
                        /torch.sum(self.weight[target]*mask_.squeeze())
        else:

            if type(self.weight) == type(None):
                loss = self.loss (pred ,target) 
            else:
                loss = self.loss (pred, target) \
                        /torch.sum(self.weight[target])
                    
        return loss 