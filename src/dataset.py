import pandas as pd
import random
import math

import torch
from torch.utils.data import Dataset
import transformers
from transformers import BertTokenizer, RobertaTokenizer, LongformerTokenizer
torch.manual_seed(4096)
random.seed(4096)
transformers.logging.set_verbosity_error()

class ArgumentMiningDataset(Dataset):
    def __init__(self, path, split, batch_size, model='bert-base-uncased'):
        self.data = pd.read_csv(f'{path}/{split}.csv', sep='\t')
        self.split = split
        self.id = []
        unique_id = self.data['id'].unique()
        if self.split == 'train':
            random.shuffle(unique_id)

        self.data_num = len(unique_id)
        for i in range(self.data_num // batch_size):
            temp = []
            for j in range(batch_size):
                temp.append(unique_id[i * batch_size + j])
            self.id.append(temp)
            

        if model in ['bert-base-uncased', 'bert-base-cased']:
            self.tokenizer = BertTokenizer.from_pretrained(model)
            self.max_leng = 512
        elif model in ['roberta-base']:
            self.tokenizer = RobertaTokenizer.from_pretrained(model)
            self.max_leng = 512
        elif model in ['allenai/longformer-base-4096']:
            self.tokenizer = LongformerTokenizer.from_pretrained(model)
            self.max_leng = 4096
        else:
            raise NotImplementedError("Not supported pretrained model type.")
        
        self.batch_size = batch_size
    
    def __len__(self):
        return self.data_num // self.batch_size

    def collate_fn(self, data):
        ID, Q, R, Q_s, Q_e, R_s, R_e = data[0]

        A = self.tokenizer(Q, R, padding=True, truncation=True, return_tensors="pt")
        A.token_type_ids[:, :2] = 1

        return ID, A, Q_s, Q_e, R_s, R_e

    def __getitem__(self, idx):
        '''
            ID: the identities of the query-response pairs (list of int)
            Q: the qeuries (list of string)
            R: the responses (list of string)
            Q_s: start indices of Q (list of int)
            Q_e: end indeices of Q (list of int)
            R_s: start indices of R (list of int)
            R_e: end indeices of R (list of int)
            TODO (1): half the batch size if its length is too long, and use window to clip Q and R
            TODO (2): valid set should not randomly choose a ground truth, but return all of the ground truth
            TODO (3): inference dataset
        '''
        Q, R, Y, Q_s, Q_e, R_s, R_e = [], [], [], [], [], [], []

        for id in self.id[idx]:
            df = self.data[self.data['id'] == id]
            
            q = df['q'].iloc[0]
            r = df['r'].iloc[0]
            y = df['s'].iloc[0]
            
            q = y + ' [SEP] ' + q
            if self.split == 'train':
                i = random.randint(0, df.shape[0] - 1)
                
                Q_s.append(df['q_s'].iloc[i])
                Q_e.append(df['q_e'].iloc[i])

                R_s.append(df['r_s'].iloc[i])
                R_e.append(df['r_e'].iloc[i])
            else:
                Q_s.append(df['q_s'].values.tolist())
                Q_e.append(df['q_e'].values.tolist())

                R_s.append(df['r_s'].values.tolist())
                R_e.append(df['r_e'].values.tolist())

            Q.append(q)
            R.append(r)

        return self.id[idx], Q, R, Q_s, Q_e, R_s, R_e

class ArgumentMiningTestDataset(Dataset):
    def __init__(self, path, batch_size, model='bert-base-uncased'):
        self.data = pd.read_csv(f'{path}/test.tsv', sep='\t')
        self.id = []
        self.data_num = self.data['id'].shape[0]

        for i in range(self.data_num // batch_size):
            temp = []
            for j in range(batch_size):
                temp.append(self.data['id'][i * batch_size + j])
            self.id.append(temp)

        temp = []
        for id in self.data['id'][- (self.data_num % batch_size):]:
            temp.append(id)
        if temp:
            self.id.append(temp)
            

        if model in ['bert-base-uncased', 'bert-base-cased']:
            self.tokenizer = BertTokenizer.from_pretrained(model)
            self.max_leng = 512
        elif model in ['roberta-base', 'xlm-roberta-base']:
            self.tokenizer = RobertaTokenizer.from_pretrained(model)
            self.max_leng = 512
        elif model in ['allenai/longformer-base-4096']:
            self.tokenizer = LongformerTokenizer.from_pretrained(model)
            self.max_leng = 4096
        else:
            raise NotImplementedError("Not supported pretrained model type.")
        
        self.batch_size = batch_size
    
    def __len__(self):
        return math.ceil(self.data_num / self.batch_size)

    def collate_fn(self, data):
        ID, Q, R = data[0]

        A = self.tokenizer(Q, R, padding=True, truncation=True, return_tensors="pt")
        A.token_type_ids[:, :2] = 1

        return ID, A

    def __getitem__(self, idx):
        Q, R, Y = [], [], []

        for id in self.id[idx]:
            df = self.data[self.data['id'] == id]
            
            q = df['q'].iloc[0]
            r = df['r'].iloc[0]
            y = df['s'].iloc[0]

            q = y + ' [SEP] ' + q
            
            Q.append(q)
            R.append(r)


        return self.id[idx], Q, R
