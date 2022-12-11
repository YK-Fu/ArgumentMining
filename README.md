# ArgumentMining

## Preporcess

Calculate the start/end indices in character-level
```
python utility/preprocess.py <path_to_raw_data> <path_to_output_data> 
```

Split data into train and valid set, output_path is the target folder
```
python utility/split_data.py <data_path> <output_path>
```

## Training
The data_path is a folder containing two preprocessed csv files named train.csv and valid.csv. Information about arguments is listed in main.py
```
python main.py \
        --name <exp_name>
        --mode train \
        --data_path <path_to_data> \
        --model <model_name> \
        --epoch <num_epoch> \
        --batch_size <batch_size> \
```

## Inference
The data_path is a folder containing raw data of testing csv files name test.csv.
```
python main.py \
        --name <exp_name>
        --mode inference \
        --data_path <path_to_data> \
        --model <model_name> \
        --ckpt <path_to_checkpoint> \
```

## Ensemble
The checkpoints should be placed in <result_path>/subset_{i}/best.ckpt, where i is the index of subset.
```
python main.py \
        --name <exp_name>
        --mode ensemble \
        --data_path <path_to_data> \
        --model <model_name> \
        --result_path <folder_to_ckpt> \
        
```
