# Copyright (c) Microsoft Corporation. Licensed under the MIT license.
#script for finetuning(MLM objective)bibert on the realcs text
BASE="/raid/speech/ashish/TSTG_new"
MODEL="xlm-roberta-large"
#MODEL="bert-base-multilingual-cased"
# PRETRAINED_MODEL="/home/ashish/benchmark/TSTG/self-training/pretrain_expts/output_ftrandom/final_model"
# PRETRAINED_TOKENIZER="/home/ashish/benchmark/TSTG/self-training/pretrain_expts/output_ftrandom"
PRETRAINED_MODEL="xlm-roberta-large"
PRETRAINED_TOKENIZER="xlm-roberta-large"
# PRETRAINED_MODEL="bert-base-multilingual-cased"
# PRETRAINED_TOKENIZER="bert-base-multilingual-cased"
MODEL_TYPE="xlmr"
# MODEL_TYPE="bert"

OUT_DIR_NAME=$1
train_file=$2
valid_file=$3

OUT_DIR="$BASE/pre-training/${OUT_DIR_NAME}"
mkdir -p $OUT_DIR

MLM_TRAIN_FILE=$train_file
echo "train file is $MLM_TRAIN_FILE"
MLM_EVAL_FILE=$valid_file

# export NVIDIA_VISIBLE_DEVICES=1
# export CUDA_VISIBLE_DEVICES=1

MLM_PROB=0.3
EPOCH=20
BATCH_SIZE=16  # set to match the exact GLUECOS repo
MAX_SEQ=256
GRAD_STEPS=5

if [ ! -d $OUT_DIR ] 
then
  mkdir -p $OUT_DIR
fi


python $BASE/pre-training/pretrain_1.py \
    --model_type $MODEL_TYPE \
    --config_name $PRETRAINED_MODEL   \
    --model_name_or_path $PRETRAINED_MODEL \
    --tokenizer_name $PRETRAINED_TOKENIZER \
    --output_dir $OUT_DIR \
    --train_data_file $MLM_TRAIN_FILE \
    --eval_data_file $MLM_EVAL_FILE \
    --mlm \
    --line_by_line \
    --do_train \
    --do_eval \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_STEPS\
    --num_train_epochs $EPOCH\
    --logging_steps 120 \
    --seed 52 \
    --save_steps 240 \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --mlm_probability $MLM_PROB


#--evaluate_during_training \
