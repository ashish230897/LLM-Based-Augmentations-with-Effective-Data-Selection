#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use turs file except in compliance with the License.
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
MODEL=${1:-xlm-roberta-large}
DATA_DIR="$REPO/data/"
OUT_DIR="$REPO/results_naacl/"

LANG=${2:-sw}
student_model=${3:-xlm-roberta-large}  # this is the initializer of the student model
OUTPUT_MODEL_NAME=${4:-sw-prompt-zeroshot-expt-2-hybrid-seed12}
AUX_TASK=${5:-snli}  #dev and test files will be read from $DATA_DIR/${AUX_TASK}
TASK=${6:-xnli}  #main task
# teacher_model=${7:-results/sst5/hisst5finetunedv2/xlm-roberta-large-LR5e-6-epoch15-MaxLen128/checkpoint-best}
teacher_model=${7:-results/snli/swsnlifinetuned/xlm-roberta-large-LR5e-6-epoch15-MaxLen128/checkpoint-best}
train_dir=${8:-data/snli/diverseK/topk/hard/}
seed=$9
# train_dir=${8:-data/lawdomain/zero-shot_randk/expt-2/}
# train_dir=${8:-data/hiproduct/diverseK/hard/}
# train_dir=${8:-data/snli/zero-shot_easy/expt-2/}

# LR=2e-5
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
  GRAD_ACC=4
  LR=5e-6
else
  BATCH_SIZE=8
  GRAD_ACC=4
  LR=2e-5
fi

SAVE_DIR="$OUT_DIR/$TASK/${OUTPUT_MODEL_NAME}/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}/"
mkdir -p $SAVE_DIR
tokenizer_name="xlm-roberta-large"

python $PWD/train_xlmr/run_classify_student_corrected_as.py \
  --model_type $MODEL_TYPE \
  --tokenizer_name $tokenizer_name \
  --model_name_or_path $student_model \
  --train_language ${LANG} \
  --dev_language ${LANG} \
  --task_name $AUX_TASK \
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
  --data_dir $DATA_DIR/${AUX_TASK} \
  --do_eval \
  --do_train \
  --temperature 1.5 \
  --seed $seed \
  --teacher_model $teacher_model \
  --train_dir $train_dir




# CUDA_VISIBLE_DEVICES=5 nohup bash train_xlmr/train_ST.sh  > logs-naacl-xnli/sw-prompt-zeroshot-expt-2-hybrid-seed12 &


#Dont forget to rename zeroshot-expt-2... to hi-zeroshot-expt-2...

