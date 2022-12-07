import pandas as pd
import random
import math

import torch
from torch.utils.data import Dataset
import transformers
from transformers import BertTokenizerFast, RobertaTokenizerFast, LongformerTokenizerFast
torch.manual_seed(4096)
random.seed(4096)
transformers.logging.set_verbosity_error()

class ArgumentMiningDataset(Dataset):
    def __init__(self, path, split, id, batch_size, model='bert-base-uncased'):
        self.data = pd.read_csv(f'{path}/sub{split}_{id}.csv', sep='\t')
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
            self.tokenizer = BertTokenizerFast.from_pretrained(model)
            self.model = 'bert'
            self.sep = ' [SEP] '
            self.offset = 8
            self.max_leng = 512
        elif model in ['roberta-base']:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(model)
            self.model = 'roberta'
            self.sep = '</s></s>'
            self.offset = 9
            self.max_leng = 512
        elif model in ['allenai/longformer-base-4096', 'allenai/longformer-large-4096']:
            self.tokenizer = LongformerTokenizerFast.from_pretrained(model)
            self.model = 'roberta'
            self.sep = '</s></s>'
            self.offset = 9
            self.max_leng = 3000
        else:
            raise NotImplementedError("Not supported pretrained model type.")
        
        self.batch_size = batch_size
    
    def __len__(self):
        return self.data_num // self.batch_size

    def convert_answer(self, offset_mapping, span):

        Q_mapped_s = -1
        Q_mapped_e = -1
        R_mapped_s = -1
        R_mapped_e = -1

        for j, o in enumerate(offset_mapping):   # [CLS] 1 [SEP] Q [SEP] R [SEP] -> Q [SEP] R [SEP]
            if o[0] == o[1] == 0:
                continue
            
            if o[0] == span[0] and Q_mapped_e < 0:
                Q_mapped_s = j
            if o[0] == span[2] and Q_mapped_e > 0 and R_mapped_e < 0:
                R_mapped_s = j
            if o[1] == span[1] and Q_mapped_s >= 0 and R_mapped_s < 0:
                Q_mapped_e = j
            if o[1] == span[3] and R_mapped_s >= 0:
                R_mapped_e = j
                break

        answer = [Q_mapped_s, Q_mapped_e, R_mapped_s, R_mapped_e]

        return answer
            
    def collate_fn(self, data):
        return data[0][0], data[0][1], data[0][2], data[0][3]

    def __getitem__(self, idx):
        '''
            ID: the identities of the query-response pairs (list of int)
            Q: the qeuries (list of string)
            R: the responses (list of string)
            S: [[q_s0, q_e0, r_s0, r_e0], [q_s1, q_e1, r_s1, r_e1], ...] the answer span for training
               [[[q_s00, q_e00, r_s00, r_e00], [q_s01, q_e01, r_s01, r_e01]], [[q_s10, q_e10, r_s10, r_e10]], ...] the answer span for validation
            TODO: half the batch size if its length is too long, and use window to clip Q and R
        '''
        A, Q, R, S = [], [], [], []

        for id in self.id[idx]:
            df = self.data[self.data['id'] == id].reset_index()
            
            q = df['q'][0]
            r = df['r'][0]
            y = df['s'][0]

            q = str(y) + self.sep + q
            Q.append(q)
            R.append(r)
            if self.split == 'train':
                i = random.randint(0, df.shape[0] - 1)
                S.append([df['q_s'][i] + self.offset, df['q_e'][i] + self.offset, df['r_s'][i], df['r_e'][i]])
            else:
                all_span = []
                for q_s, q_e, r_s, r_e in df[['q_s', 'q_e', 'r_s', 'r_e']].values.tolist():
                    all_span.append([q_s + self.offset, q_e + self.offset, r_s, r_e])
                S.append(all_span)

        A = self.tokenizer(Q, R, 
                padding='longest', 
                max_length=self.max_leng, 
                truncation="longest_first", 
                return_overflowing_tokens=False, 
                return_offsets_mapping=True,
                return_tensors='pt'
            )

        if self.model == 'bert':
            A.token_type_ids[:, :2] = 1
        
        TS = []
        offset_mapping = A.pop('offset_mapping')
        if self.split == 'train':
            for id, mapping, s in zip(self.id[idx], offset_mapping, S):
                ts = self.convert_answer(mapping, s)
                
                if ts[0] == ts[1] == -1 or ts[2] == ts[3] == -1:
                    ts = [-1] * 4
                TS.append(ts)
                
            TS = torch.LongTensor(TS)
        else:
            for mapping , all_s in zip(offset_mapping, S):
                temp_s = []
                for s in all_s:
                    ts = self.convert_answer(mapping, s)
                    if ts[0] == ts[1] == -1 or ts[2] == ts[3] == -1:
                        ts = [-1] * 4
                    temp_s.append(ts)

                TS.append(temp_s)
            TS = [torch.LongTensor(ts) for ts in TS]
        
        
        return self.id[idx], A, TS, offset_mapping

class ArgumentMiningTestDataset(Dataset):
    def __init__(self, path, batch_size, model='bert-base-uncased'):
        self.data = pd.read_csv(f'{path}/test.csv', sep=',')
        
        self.data['q'].apply(lambda x: x.strip('\"').strip(' '))
        self.data['r'].apply(lambda x: x.strip('\"').strip(' '))

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
            
        self.data_num = len(self.data['id'])
        for i in range(self.data_num // batch_size):
            temp = []
            for j in range(batch_size):
                temp.append(self.data['id'][i * batch_size + j])
            self.id.append(temp)
            

        if model in ['bert-base-uncased', 'bert-base-cased']:
            self.tokenizer = BertTokenizerFast.from_pretrained(model)
            self.model = 'bert'
            self.sep = ' [SEP] '
            self.offset = 8
            self.max_leng = 512
        elif model in ['roberta-base']:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(model)
            self.model = 'roberta'
            self.sep = '</s></s>'
            self.offset = 9
            self.max_leng = 512
        elif model in ['allenai/longformer-base-4096', 'allenai/longformer-large-4096', 'valhalla/longformer-base-4096-finetuned-squadv1']:
            self.tokenizer = LongformerTokenizerFast.from_pretrained(model)
            self.model = 'roberta'
            self.sep = '</s></s>'
            self.offset = 9
            self.max_leng = 4096
        else:
            raise NotImplementedError("Not supported pretrained model type.")
        
        self.batch_size = batch_size
    
    def __len__(self):
        return math.ceil(self.data_num / self.batch_size)

    def collate_fn(self, data):
        return data[0][0], data[0][1], data[0][2]

    def __getitem__(self, idx):
        '''
            Q: the qeuries (list of string)
            R: the responses (list of string)
            TODO: half the batch size if its length is too long, and use window to clip Q and R
        '''
        A, Q, R = [], [], []

        for id in self.id[idx]:
            df = self.data[self.data['id'] == id].reset_index()
            
            q = df['q'][0]
            r = df['r'][0]
            y = 1 if df['s'][0] == 'AGREE' else 0

            q = str(y) + self.sep + q
            Q.append(q)
            R.append(r)

        A = self.tokenizer(Q, R, 
                padding='longest', 
                max_length=self.max_leng, 
                truncation="longest_first", 
                return_overflowing_tokens=False, 
                return_offsets_mapping=True,
                return_tensors='pt'
            )

        if self.model == 'bert':
            A.token_type_ids[:, :2] = 1
        offset_mapping = A.pop('offset_mapping')
        
        return self.id[idx], A, offset_mapping

