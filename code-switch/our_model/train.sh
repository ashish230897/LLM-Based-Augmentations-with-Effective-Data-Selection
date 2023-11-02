curr_dir=${PWD}

# model args
model_name="bigscience/mt0-xl"
device_map="auto"
lora_alpha=32
lora_dropout=0.05
lora_r=16

# data args
max_seq_length_input=256
max_seq_length_output=256
train_file_path="${curr_dir}/data/train.csv"
valid_file_path="${curr_dir}/data/valid.csv"

# training args
per_device_train_batch_size=16
gradient_accumulation_steps=4  # making the effective batch size as 64
per_device_eval_batch_size=32
save_steps=50
eval_steps=50
logging_steps=1
learning_rate=2e-4
max_grad_norm=0.3
warmup_ratio=0.03
lr_scheduler_type="constant"
num_train_epochs=15
save_total_limit=1
output_dir="${curr_dir}/models/mt0xlv1/"
evaluation_strategy="steps"
metric_for_best_model="eval_loss"
num_beams=1
report_to="wandb"
wandb_project="mt0-cs-training"

# --fp16 \
# --group_by_length \
# --optim $optim \
# --load_in_8bit \

python ${curr_dir}/train.py \
    --model_name_or_path $model_name \
    --device_map $device_map \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --lora_r $lora_r \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --save_steps $save_steps \
    --eval_steps $eval_steps \
    --logging_steps $logging_steps \
    --learning_rate $learning_rate \
    --max_grad_norm $max_grad_norm \
    --warmup_ratio $warmup_ratio \
    --lr_scheduler_type $lr_scheduler_type \
    --num_train_epochs $num_train_epochs \
    --max_seq_length_input $max_seq_length_input \
    --max_seq_length_output $max_seq_length_output \
    --train_file_path $train_file_path \
    --valid_file_path $valid_file_path \
    --load_best_model_at_end \
    --save_total_limit $save_total_limit \
    --output_dir $output_dir \
    --evaluation_strategy $evaluation_strategy \
    --metric_for_best_model $metric_for_best_model \
    --report_to $report_to \
    --wandb_project $wandb_project \
    #--predict_with_generate \
    #--generation_num_beams $num_beams \
    #--generation_max_length $max_seq_length_output \
    #--include_inputs_for_metrics 
