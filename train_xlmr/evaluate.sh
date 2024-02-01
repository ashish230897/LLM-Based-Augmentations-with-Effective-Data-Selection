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
TASK=$4
seed=$5

LR=2e-5
EPOCH=15
# EPOCH=1
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
  BATCH_SIZE=5
  GRAD_ACC=6
  #GRAD_ACC=2
  #LR=3e-5
  LR=5e-6
else
  BATCH_SIZE=8
  GRAD_ACC=4
  LR=2e-5
fi

SAVE_DIR=$saved_model

tokenizer_name="xlm-roberta-large"

python $PWD/train_xlmr/run_classify.py \
  --model_type $MODEL_TYPE \
  --tokenizer_name $tokenizer_name \
  --model_name_or_path $saved_model \
  --train_language ${LANG} \
  --dev_language ${LANG} \
  --task_name $TASK \
  --do_eval \
  --data_dir $DATA_DIR/${TASK} \
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
  --seed $seed \
  --do_predict \

# command
# bash train_xlmr/evaluate.sh xlm-roberta-large hi /raid/speech/ashish/TSTG_new/results/hiproduct/fewshot-expt-3-topk-t1.5-seed42/xlm-roberta-large-LR5e-6-epoch15-MaxLen128/checkpoint-best hiproduct 42