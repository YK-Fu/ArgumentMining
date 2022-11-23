import pandas as pd
import torch
import random
from torch.utils.data import Dataset
from transformers import BertTokenizer, RobertaTokenizer, LongformerTokenizer
torch.manual_seed(4096)
random.seed(4096)

class ArgumentMining(Dataset):
    def __init__(self, path, split, batch_size, model='bert-base-uncased'):
        self.data = pd.read_csv(f'{path}/{split}.csv', sep='\t')
        self.id = self.data['id'].unique()

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
        return len(self.id) // self.batch_size

    def collate_fn(self, data):
        ID, Q, R, Y, Q_s, Q_e, R_s, R_e = data[0]

        Q = self.tokenizer.batch_encode_plus(Q, padding=True, truncation=True, return_tensors="pt")
        R = self.tokenizer.batch_encode_plus(R, padding=True, truncation=True, return_tensors="pt")
        Y = torch.FloatTensor(Y).unsqueeze(-1)
        
        Q_s = torch.LongTensor(Q_s)
        Q_e = torch.LongTensor(Q_e)
        R_s = torch.LongTensor(R_s)
        R_e = torch.LongTensor(R_e)

        return ID, Q, R, Y, Q_s, Q_e, R_s, R_e

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
        ID, Q, R, Y, Q_s, Q_e, R_s, R_e = [], [], [], [], [], [], [], []

        for id in self.id[idx: idx + self.batch_size]:
            df = self.data[self.data['id'] == id]
            
            q = df['q'].iloc[0]
            r = df['r'].iloc[0]
            y = df['s'].iloc[0]
            
            i = random.randint(0, df.shape[0] - 1)
            
            Q_s.append(df['q_s'].iloc[i])
            Q_e.append(df['q_e'].iloc[i])

            R_s.append(df['r_s'].iloc[i])
            R_e.append(df['r_e'].iloc[i])

            Q.append(q)
            R.append(r)
            Y.append(y)
            ID.append(id)
            
        return ID, Q, R, Y, Q_s, Q_e, R_s, R_e
