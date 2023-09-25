#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
DATA_DIR="$REPO/data/"
OUT_DIR="$REPO/results/"

LANG=$2
saved_model=$3
OUTPUT_MODEL_NAME=$4
TASK=$5

LR=2e-5
EPOCH=15 #15 10 50
MAXL=128
LANGS="${LANG}"
LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
fi

if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=32
  GRAD_ACC=6
  #GRAD_ACC=2
  #LR=3e-5
  LR=5e-6
else
  BATCH_SIZE=8
  GRAD_ACC=4
  LR=2e-5
fi

SAVE_DIR="$OUT_DIR/$TASK/${OUTPUT_MODEL_NAME}/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}/"
mkdir -p $SAVE_DIR

tokenizer_name="xlm-roberta-large"
#saved_model="/raid/speech/ashish/TSTG_new/pre-training/output_syn${SEED}${LANG}2000subset/final_model/"
# saved_model="/raid/speech/ashish/TSTG_new/pre-training/output_syn${SEED}en2000subset/final_model/"
# saved_model="xlm-roberta-large"

#--do_train \
  #--do_eval \

# 
# CUDA_VISIBLE_DEVICES=1 python $PWD/train_xlmr/run_classify_student_ST2.py \
# CUDA_VISIBLE_DEVICES=3 


# python $PWD/train_xlmr/run_classify_student_ST2.py \
# python $PWD/train_xlmr/run_classify_regular.py \
# python $PWD/train_xlmr/run_classify_regular.py \
python $PWD/train_xlmr/run_classify_student_ST1.py \
  --model_type $MODEL_TYPE \
  --tokenizer_name $tokenizer_name \
  --model_name_or_path $saved_model \
  --train_language ${LANG} \
  --dev_language ${LANG} \
  --task_name $TASK \
  --do_predict \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --max_seq_length $MAXL \
  --output_dir $SAVE_DIR \
  --save_steps 100 \
  --eval_all_checkpoints \
  --log_file 'train' \
  --predict_languages $LANGS \
  --save_only_best_checkpoint \
  --overwrite_output_dir \
  --eval_test_set $LC \
  --overwrite_cache \
  --data_dir $DATA_DIR/${TASK} \
--do_train \
  --do_eval \
  --temperature 1.5  

  




# bash train_xlmr/train_ST.sh xlm-roberta-large bn xlm-roberta-large selftrain-1a-s1-ep1 bnsentiment    #SC1

# bash train_xlmr/train.sh xlm-roberta-large bn xlm-roberta-large selftrain-2 bnsentiment    #SC2


# LANG=$2 bn ?
# saved_model=$3 "xlm-roberta-large"
# OUTPUT_MODEL_NAME=$4 selftrain-1
# TASK=$5 bnsentiment ?

# bash train_xlmr/train_ST.sh xlm-roberta-large bn xlm-roberta-large selftrain-1b-t11 bnsentiment    

# bash train_xlmr/train_ST.sh xlm-roberta-large bn xlm-roberta-large regular_check-1ep bnsentiment    #SC2

# 1. Regular finetuning on bn target: 
# CUDA_VISIBLE_DEVICES=0 bash train_xlmr/train_ST.sh xlm-roberta-large bn xlm-roberta-large bnfinetuning-1 bnsentiment

# CUDA_VISIBLE_DEVICES=4 bash train_xlmr/train_ST.sh xlm-roberta-large bn xlm-roberta-large selftrain-1b-t2 bnsentiment
# CUDA_VISIBLE_DEVICES=0 bash train_xlmr/train_ST.sh xlm-roberta-large bn xlm-roberta-large selftrain-2a-s1 bnsentiment

# CUDA_VISIBLE_DEVICES=5 bash train_xlmr/train_ST.sh xlm-roberta-large bn xlm-roberta-large selftrain-1b-s1 bnsentiment

# CUDA_VISIBLE_DEVICES=0 bash train_xlmr/train_ST.sh xlm-roberta-large bn xlm-roberta-large selftrain-1a-s1-redone bnsentiment
# CUDA_VISIBLE_DEVICES=5 bash train_xlmr/train_ST.sh xlm-roberta-large bn xlm-roberta-large selftrain-1b-s1-redone bnsentiment

# CUDA_VISIBLE_DEVICES=3 bash train_xlmr/train_ST.sh xlm-roberta-large bn xlm-roberta-large selftrain-1b-s3-t2 bnsentiment    #SC2
# CUDA_VISIBLE_DEVICES=3 bash train_xlmr/train_ST.sh xlm-roberta-large bn xlm-roberta-large exp-S-2.5k-sst-t1.5 bnsentiment
# CUDA_VISIBLE_DEVICES=1 nohup bash train_xlmr/train_ST.sh xlm-roberta-large bn xlm-roberta-large exp-S-5k-sst-t0.5 bnsentiment > log-S-5-t0.5 &


# CUDA_VISIBLE_DEVICES=5 nohup bash train_xlmr/train_ST.sh xlm-roberta-large bn xlm-roberta-large exp-B-5k-sst-t1.5 bnsentiment > log-B-5-t1.5 &
# CUDA_VISIBLE_DEVICES=5 nohup bash train_xlmr/train_ST.sh xlm-roberta-large bn xlm-roberta-large exp-B-2.5k-sst-t1.5 bnsentiment > log-B-2.5-t1.5 &
# CUDA_VISIBLE_DEVICES=0 nohup bash train_xlmr/train_ST.sh xlm-roberta-large bn xlm-roberta-large exp-CC-2.5k-sst-t1.5 bnsentiment > log-C-2.5-t1.5 &
