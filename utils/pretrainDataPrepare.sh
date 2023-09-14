#!/bin/sh

input_path=$1
task=$2
type=$3

if [ $type == "target" ]; then
    python utils/createTargetPretrainData.py --task $task --input_path $input_path
else
    echo "Landed here"
    python utils/createPretrainData.py --task $task --path $input_path
fi

python utils/split_pretrain.py --file_path "${input_path}.pretrain"

python pre-training/create_data.py --train_path "${input_path}.pretrain.train" --valid_path "${input_path}.pretrain.valid" --model xlm-roberta-large

