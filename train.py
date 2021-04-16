import numpy as np
np.random.seed(1234)
import os 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

import argparse
import time
import pickle
import os

from pathlib import Path 
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,\
                        classification_report, precision_recall_fscore_support


from joblib import Parallel, delayed
import multiprocessing

from models import MULTModel, MaskedLoss
from dataloader import TPOT_loader
#from model import SimpleModel, MaskedLoss #, BERT_emotion_classifier
#from model_lmf import LMF, MaskedLoss
#from transformers import AdamW, get_linear_schedule_with_warmup


from utilities import *


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_TPOT_loaders(domain, data,  fold=0,  batch_size=32, valid=0.1,  indices = None, num_workers=0, pin_memory=False):
    data_dir = data
    #[tr_indices, vl_indices, ts_indices] = indices
    tr_indices = os.path.join(data_dir, 'train_'+ str(fold) +'.npy')
    vl_indices = os.path.join(data_dir, 'valid_'+ str(fold) +'.npy')
    ts_indices = os.path.join(data_dir, 'test_'+ str(fold) +'.npy')
    
    #----------Pre-trained Featuures loading------------#
    #tr_feat = os.path.join(feature_dir, 'train_'+ str(fold) +'.npy')
    #vl_feat = os.path.join(feature_dir, 'valid_'+ str(fold) +'.npy')
    #ts_feat = os.path.join(feature_dir, 'test_'+ str(fold) +'.npy')
   
    trainset = TPOT_loader(full_data = tr_indices, select_indices = None, domain= domain)
    
   

    #train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              #sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    
    validset = TPOT_loader(full_data = vl_indices, select_indices =None, domain= domain,scaler= None, train=False)
   
    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              shuffle=False,
                              #sampler=valid_sampler,
                              collate_fn=validset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    
    testset = TPOT_loader(full_data = ts_indices, select_indices = None, domain= domain,scaler = None, train=False)

    test_loader = DataLoader(testset,
                             shuffle=False,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    
    
   
    #return train_loader, valid_loader, test_loader
    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, device=None, optimizer=None, train=False):
    losses = []
    preds = []
    labels = []
    masks = []
    all_probs=[]
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    
    for data in dataloader:
        if train:
            optimizer.zero_grad()
     
        audio, video, text, label, a_mask, v_mask, t_mask =\
                [d.to(device) for d in data[:-1]] if device is not None else data[:-1]
        

        '''
        text, bert_mask, label = [d.to(device) for d in data[:-1]] if device is not None else data[:-1]
        '''
        names = data[-1]
      
     
        umask = torch.LongTensor ([1]*len(text))
        
        log_prob, _ = model (text, audio, video)
        

        lp_ = log_prob.view(-1,log_prob.size()[1])
        labels_= label.view(-1) 

        
        probabilities = torch.exp(lp_)
      
        loss = loss_function(lp_, labels_)
        
        #print (loss.item())
        
        '''
        pred_ = torch.argmax(lp_,1) # batch*seq_len
        probs = torch.softmax (lp_,1)
        preds.append(pred_.data.cpu().numpy())
        all_probs.append(probs.data.cpu().numpy())
        
        labels.append(labels_.data.cpu().numpy())
        '''
        pred_ = torch.argmax(probabilities,1)
        preds.append (pred_.data.cpu().numpy())
        labels.append (labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        all_probs.append (probabilities.data.cpu().numpy())
        
        losses.append(loss.item()*masks[-1].sum())
        #losses.append (loss.item()*len(masks).sum())
       
        # print (loss.item())

        if np.isnan (loss.item()):
            pdb.set_trace()
        #print (loss)
        if train:

            loss.backward()
            
            optimizer.step()
        else:
           # alphas += alpha
            #alphas_f += alpha_f
            #alphas_b += alpha_b
            vids += data[-1]
        
    
    if preds!=[]:
        
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
        all_probs=  np.concatenate (all_probs)
        
    else:
        return float('nan'), float('nan'), [], [], [], float('nan') #,[]
    

    avg_loss = round(np.sum(losses)/np.sum(masks),4)
    
    avg_accuracy = round(accuracy_score(labels,preds,sample_weight=masks)*100,2)
    avg_fscore = round(f1_score(labels,preds,sample_weight=masks,average='weighted')*100,2)
    
   
    return avg_loss, avg_accuracy, labels, preds, masks,avg_fscore ,all_probs#, [alphas, alphas_f, alphas_b, vids]

class params:

    def __init__(self):
        self.orig_d_l = 300 
        self.orig_d_a = 138
        self.orig_d_v = 68
        self.output_dim = 2

if __name__ == '__main__':

    hyp_params = params()

    
    fold_dir = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/kfold_splits/'   
    data_dir = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/tpot_multi_data/'
    feature_dir = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/feature_both/'
    '''
    full_data = combine_splits(data_dir)
    np.save (os.path.join (data_dir, 'FULL_DATA.npy'), full_data)
    '''
    
    #full_data = np.load (os.path.join(data_dir,'FULL_DATA.npy'), allow_pickle=True).item()
    #list_ = np.load ('names.npy', allow_pickle=True)

    #tr, vl, ts = get_k_fold (list_, n_folds = 10)
    #--------Enumerate for the folds ----------#
    '''
    tr_l,vl_l,ts_l = [], [],[]
    for idx in range(10):
        tr, vl , ts = get_split_indices (fold_dir, full_data)
        
        tr_l.append (tr)
        vl_l.append (vl)
        ts_l.append (ts)


    "Uncomment and use this only once . It will help in faster debugging in training. Otherwise the loading is computationally extensively"
    #Crreate fold data 
    def make_folds (f_id, tr_indices, vl_indices, ts_indices):
    

        data = dict ()
        for keys, values in full_data.items():        
            data[keys] = np.array( values)[tr_indices]

        np.save(os.path.join(data_dir, 'train_'+str(f_id)+'.npy'), data)
        data = dict ()
        for keys, values in full_data.items():
            data[keys] = np.array( values)[vl_indices]
        np.save(os.path.join(data_dir, 'valid_'+str(f_id)+'.npy'), data)
        data = dict ()
        for keys, values in full_data.items():
            data[keys] = np.array( values)[ts_indices]
        np.save(os.path.join(data_dir, 'test_'+str(f_id)+'.npy'), data)
    
    _= Parallel(n_jobs=1)(delayed(make_folds)(f_id,tr_indices, vl_indices, ts_indices) for f_id,(tr_indices, vl_indices, ts_indices) in enumerate(zip (tr_l,vl_l,ts_l)))
    pdb.set_trace()
    '''
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument ('--domain', default='Positive', help='Enter a proper emotion / strategy')
    parser.add_argument ('--modal', default='all', help='Enter a proper emotion')
    args = parser.parse_args()

    print(args)
    
    if args.domain not in ['Positive','Aggressive', 'Dysphoric','strategy']:
        pdb.set_trace()
    else:
        save_dir= './result_' + args.domain + '_' + args.modal + '/'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        domain = args.domain


    #glove_matrix = np.load (os.path.join('/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/','glv_embedding_matrix_apo'),allow_pickle=True)
    #glove_matrix = np.load (os.path.join('/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/Sanchayan','glv_evmbedding_matrix'),allow_pickle=True)
    
    embedding_path = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/tpot_multi_data'
    glove_matrix = np.load (os.path.join(embedding_path,'glv_embedding_matrix'),allow_pickle=True)
    #print (os.listdir(file_path))
    
    "Cuda related Operations"    
    cuda = torch.cuda.is_available() 
    #args.cuda = False
  
    if cuda:
        print('Running on GPU')
        device = torch.device ('cuda:3')
    else:
        print('Running on CPU')
        device = torch.device ('cpu')
    #device = torch.device('cpu')

    batch_size= 32
    n_classes = 2
    n_epochs=15


    if domain == 'Positive':
        D_a = 57
        D_v = 43
        #858 vs 9005
        class_weights = [1/0.875,1/0.125]
    elif domain == 'Dysphoric':
        D_a = 68
        D_v = 32
        #4125 vs 5738
        class_weights = [1/0.61,1/0.39]
    elif domain == 'Aggressive':
        D_a = 54
        D_v = 46
        #954 vs 8909
        class_weights = [1/0.90,1/0.10]
    elif domain == 'strategy':
        vocab_size = glove_matrix.shape[0]
        D_a = 138 
        D_v = 236
        D_t = 300
        H_a = 512
        H_v = 512
        H_t = 512

        class_weights = [1/0.65, 1/0.35]
        #class_weights= None 
    vocab_size = glove_matrix.shape[0]
    embedding_dim=300
   
    
    emo_dim=2 
    speaker_dim=2
    class_hid = 512



    config_list =['0.001_0.0',
                    '0.001_0.3',
                    '0.001_0.5',
                    '0.0005_0.0',
                    '0.0005_0.3',
                    '0.0005_0.5',
                    '0.0005_0.8']
    

    '''
    config_list = ['0.001_0.3', 
                   '0.0001_0.3',
                   '0.01_0.5',
                   '0.001_0.5',
                   '0.01_0.3', 
                   '0.0001_0.5']
    '''
    for f_id in range(10):

        train_loader, valid_loader, test_loader =\
                 get_TPOT_loaders(domain= domain, 
                    data = data_dir,
                    fold=f_id,
                    valid=0.1,
                    #indices = [tr_indices, vl_indices, ts_indices],
                    batch_size=batch_size,
                    num_workers=0)

 
        for config in config_list:
            lr_config, drop_config= config.split('_')
            

            #-----Data preparation -------#
                      
            #model = BERT_emotion_classifier (n_classes= n_classes, dropout=float(drop_config))
            
            #submodel_params = [D_a, D_v, vocab_size, embedding_dim, gap_dim, H_a, H_v, H_t, H_g, emo_dim, speaker_dim, class_hid, n_classes, args.modal, device, glove_matrix]
            #model= LMF (submodel_params, output_dim = 512, rank=4, use_softmax=True, dropout=float(drop_config))
            
            #model = SimpleModel (D_a , D_v, vocab_size, embedding_dim, H_a, H_v, H_t , class_hid, n_classes=2, modal =args.modal,  \
            #                     drop= float(drop_config) , device = device)
            model = MULTModel (hyp_params, vocab_size = vocab_size,drop = float(drop_config)) 
            model.init_pretrained_embeddings_from_numpy (glove_matrix)
            model.to(device)
           
            
            loss_weights = torch.FloatTensor(class_weights).to(device)
            
            #loss_weights = torch.FloatTensor([])
            '''
            if args.class_weight:
                loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
            else:
                loss_function = MaskedNLLLoss()
            '''
            #---------------Change the loss function to BCE Loss -------------------#
            #loss_function = MaskedNLLLoss(weight=loss_weights).to(device)
            loss_function = MaskedLoss (weight=loss_weights).to(device)
            optimizer= optim.Adam(model.parameters(), lr=float(lr_config), weight_decay = 0.01)
            
            
                  
            best_loss, best_label, best_pred, best_mask, best_probs = None, None, None, None, None
            best_epoch = None 

            train_losses=[]
            valid_losses=[]
            for e in range(n_epochs):
                start_time = time.time()
            
                
                train_loss, train_acc, _,_,_,train_fscore, train_probs= train_or_eval_model(model, loss_function,train_loader, e, optimizer=optimizer,device=device, train= True)
                valid_loss, valid_acc, _,_,_,val_fscore, val_probs= train_or_eval_model(model, loss_function, valid_loader, e,device=device)
                
                test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, test_probs= train_or_eval_model(model, loss_function, test_loader, e,device=device)
            
                train_losses.append(train_loss)
                valid_losses.append(valid_loss) 
            
                if best_loss == None or best_loss > test_loss:
                    '''
                    best_loss, best_label, best_pred, best_mask, best_attn =\
                            test_loss, test_label, test_pred, test_mask, attentions
                    '''
                    best_loss, best_label, best_pred, best_mask, best_probs = test_loss, test_label, test_pred, test_mask, test_probs
                    best_epoch = e 
            

                '''
                if args.tensorboard:
                    writer.add_scalar('test: accuracy/loss',test_acc/test_loss,e)
                    writer.add_scalar('train: accuracy/loss',train_acc/train_loss,e)
                '''
                print('epoch {} train_loss {} train_acc {} train_fscore{} valid_loss {} valid_acc {} val_fscore{} test_loss {} test_acc {} test_fscore {} time {}'.\
                        format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, val_fscore,\
                                 test_loss, test_acc, test_fscore, round(time.time()-start_time,2)))
        
            '''
            if args.tensorboard:
                writer.close()
            '''
            print('Test performance..')
            print('Loss {} accuracy {}'.format(best_loss,
                                             round(accuracy_score(best_label,best_pred,sample_weight=best_mask)*100,2)))
            c_report=classification_report(best_label,best_pred,sample_weight=best_mask,digits=4)
            #c_mat = confusion_matrix(best_label,best_pred,sample_weight=best_mask)
            
            print (c_report)
            print (str(f_id) + '_'+ config, str(best_epoch))            

            # with open('best_attention.p','wb') as f:
            #     pickle.dump(best_attn+[best_label,best_pred,best_mask],f)
            
            #Write 
            
            res={'classification_report': c_report,\
                 #'confusion_matrix': c_mat,\
                 'train_loss': np.array(train_losses),\
                 'valid_losses': np.array(valid_losses),\
                 'test_loss': best_loss,\
                 'labels': best_label,\
                 'prediction':best_pred,\
                 'mask': best_mask,\
                 'probs': best_probs}

            np.save (save_dir+str(f_id)+'_'+config+'xresult.npy', res)