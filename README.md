# ArgumentMining

## Preporcess

Calculate the start/end indices given BERT tokenizer (or other pretrained LM tokenizer)
```
cd utility
python preprocess.py --model <model_name> --data_path <path_to_raw_data> --out <path_to_output_data> 
```

Split data into train and valid set
```
cd utility
python split_data.py <data_path> <output_path>
```

## Training
Information about arguments is listed in main.py, note that the model name must be the same with that you used in preprocess.
```
python main.py \
        --mode train \
        --data_path <path_to_data> \
        --model <model_name> \
        --epoch <num_epoch> \
        --batch_size <batch_size> \
        --grad_steps <gradient_accumulation_steps> \
        --optim AdamW,0.0001,1500,5000 \
```

## Inference
TODO
