import sys
import pandas as pd
import csv
'''
    Preprocess start and end indices of given tokenizer (character level)
    TODO: clean the input data
'''
if __name__ == '__main__':
    data_path, output_path = sys.argv[1], sys.argv[2]
    
    raw = pd.read_csv(data_path, sep=',')
    
    raw['s'] = raw['s'].apply(lambda x: 1 if x == 'AGREE' else 0)
    
    for k in ['q', 'r', 'q\'', 'r\'']:
        raw[k] = raw[k].apply(lambda x: ' ' + x.strip('\"').strip(' ') + ' ')

    q_start = []
    q_end = []
    r_start = []
    r_end = []
    
    to_drop = []

    for i, (q, r, q_, r_) in enumerate(zip(raw['q'], raw['r'], raw['q\''], raw['r\''])):
        q_s = q.find(q_)
        r_s = r.find(r_)
        
        q_e = q_s + len(q_) - 2
        r_e = r_s + len(r_) - 2

        if q_s < 0 or r_s < 0:
            to_drop.append(i)

        q_start.append(q_s)
        q_end.append(q_e)
        r_start.append(r_s)
        r_end.append(r_e)
    for k in ['q', 'r']:
        raw[k] = raw[k].apply(lambda x: x.strip(' '))
    raw['q_s'] = q_start
    raw['q_e'] = q_end
    raw['r_s'] = r_start
    raw['r_e'] = r_end

    raw.drop(to_drop, axis=0, inplace=True)
    raw[['id', 'q', 'r', 'q_s', 'q_e', 'r_s', 'r_e', 's']].to_csv(output_path, sep='\t', index=False, quoting=csv.QUOTE_NONE)
