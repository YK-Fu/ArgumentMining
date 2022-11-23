import argparse
import pandas as pd
from transformers import BertTokenizer
import csv
'''
    Preprocess start and end indices of given tokenizer
    TODO: clean the input data
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Model to tokenize")
    parser.add_argument("--data_path", type=str, default="", help="Path to a data folder containing [train.csv, valid.csv, test.csv]")
    parser.add_argument("--out", type=str, default="", help="")
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model)
    raw = pd.read_csv(args.data_path, sep='\t')

    for k in ['q', 'r']:
        raw[k].apply(lambda x: ' '.join(map(str, tokenizer.encode(x, truncation=True))))
    for k in [ 'q\'', 'r\'']:
        raw[k].apply(lambda x: ' '.join(map(str, tokenizer.encode(x, truncation=True, add_special_tokens=True))))
    q_start = []
    q_end = []
    r_start = []
    r_end = []
    
    to_drop = []

    for i, (q, r, q_, r_) in enumerate(zip(raw['q'], raw['r'], raw['q\''], raw['r\''])):
        q_s = q.find(q_)
        r_s = r.find(r_)
        
        if q_s < 0 or r_s < 0:
            to_drop.append(i)
        
        q_s = len(q[:q_s].split())
        if q_s == 0:
            q_s = 1
        q_e = q_s + len(q_.split()) - 1

        r_s = len(r[:r_s].split())
        if r_s == 0:
            r_s = 1
        r_e = r_s + len(r_.split()) - 1

        q_start.append(q_s)
        q_end.append(q_e)
        r_start.append(r_s)
        r_end.append(r_e)
    
    raw['q_s'] = q_start
    raw['q_e'] = q_end
    raw['r_s'] = r_start
    raw['r_e'] = r_end

    raw.drop(to_drop, axis=0, inplace=True)
    raw[['id', 'q', 'r', 'q_s', 'q_e', 'r_s', 'r_e', 's']].to_csv(args.out, sep='\t', index=False, quoting=csv.QUOTE_NONE)

        