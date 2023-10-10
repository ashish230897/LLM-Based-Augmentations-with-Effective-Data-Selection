curr_dir=${PWD}

# model args
model_name="xlm-roberta-large"
mlm_probability=0.3

# data args
max_length=256
pretrain_train_file_path="${curr_dir}/data/bnsentiment/pretraining/pretrain_label_en_cleanv3.csv.pretrain.train.proc"
pretrain_valid_file_path="${curr_dir}/data/bnsentiment/pretraining/pretrain_label_en_cleanv3.csv.pretrain.valid.proc"
sentiment_train_file_path="${curr_dir}/data/bnsentiment/train-en.tsv"
sentiment_valid_file_path="${curr_dir}/data/bnsentiment/dev-en.tsv"
sentiment_test_file_path="${curr_dir}/data/bnsentiment/test-en.tsv"

# training args
output_dir="${curr_dir}/results/multitask/pretrainbnsentiment/"
learning_rate=5e-6
per_device_train_batch_size=5
gradient_accumulation_steps=6  # making the effective batch size as 30
per_device_eval_batch_size=16
save_steps=1
eval_steps=1
logging_steps=1
max_grad_norm=1
lr_scheduler_type="linear"
num_train_epochs=15
save_total_limit=1
evaluation_strategy="steps"
metric_for_best_model="eval_loss"


python ${curr_dir}/multi-task-training/train.py \
    --model_name_or_path $model_name \
    --mlm_probability $mlm_probability \
    --max_length $max_length \
    --pretrain_train_file_path $pretrain_train_file_path \
    --pretrain_valid_file_path $pretrain_valid_file_path \
    --sentiment_test_file_path $sentiment_test_file_path \
    --sentiment_train_file_path $sentiment_train_file_path \
    --sentiment_valid_file_path $sentiment_valid_file_path \
    --output_dir $output_dir \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --save_steps $save_steps \
    --eval_steps $eval_steps \
    --logging_steps $logging_steps \
    --learning_rate $learning_rate \
    --max_grad_norm $max_grad_norm \
    --lr_scheduler_type $lr_scheduler_type \
    --num_train_epochs $num_train_epochs \
    --load_best_model_at_end \
    --save_total_limit $save_total_limit \
    --evaluation_strategy $evaluation_strategy \
    --metric_for_best_model $metric_for_best_model \