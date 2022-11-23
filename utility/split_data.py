import pandas as pd
import sys
import random
random.seed(4096)

# Split data into train and valid set

if __name__ == '__main__':
    data_path, out_path = sys.argv[1], sys.argv[2]

    df = pd.read_csv(data_path, sep='\t')

    ids = df['id'].unique()
    random.shuffle(ids)

    train_ids = ids[:int(len(ids) * 0.9)]
    valid_ids = ids[int(len(ids) * 0.9):]

    train_df = df[df['id'].isin(train_ids)]
    valid_df = df[df['id'].isin(valid_ids)]

    train_df.to_csv(f'{out_path}/train.csv', sep='\t', index=False)
    valid_df.to_csv(f'{out_path}/valid.csv', sep='\t', index=False)