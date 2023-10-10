curr_dir=${PWD}

# model args, taking lora config from huggingface repo: https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing#scrollTo=Ybeyl20n3dYH
model_name="meta-llama/Llama-2-7b-hf"
device_map="auto"
lora_alpha=16
lora_dropout=0.5
lora_r=8
# lora_alpha=4
# lora_dropout=0.7
# lora_r=4
bnb_4bit_quant_type="nf4"

# data args
max_seq_length=128  # this param depends on the size of texts in the fine-tuning data
file_path="${curr_dir}/data/sst5/train-en_texts.csv"

# training args
per_device_train_batch_size=16
gradient_accumulation_steps=4
optim="paged_adamw_8bit"  # following hf repo hyper params
save_steps=20
eval_steps=20
logging_steps=1
learning_rate=2e-4
max_grad_norm=0.3
warmup_ratio=0.03
lr_scheduler_type="constant"
num_train_epochs=5
output_dir="${curr_dir}/results/llama2/sst5_srno4/"
save_total_limit=1
evaluation_strategy="steps"
metric_for_best_model="eval_loss"

# --group_by_length \

python ${curr_dir}/train_llama2/tune_llama2.py \
    --model_name_or_path $model_name \
    --bnb_4bit_quant_type $bnb_4bit_quant_type \
    --device_map $device_map \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --lora_r $lora_r \
    --output_dir $output_dir \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --save_steps $save_steps \
    --eval_steps $eval_steps \
    --logging_steps $logging_steps \
    --learning_rate $learning_rate \
    --max_grad_norm $max_grad_norm \
    --warmup_ratio $warmup_ratio \
    --lr_scheduler_type $lr_scheduler_type \
    --max_seq_length $max_seq_length \
    --num_train_epochs $num_train_epochs \
    --load_best_model_at_end \
    --save_total_limit $save_total_limit \
    --evaluation_strategy $evaluation_strategy \
    --metric_for_best_model $metric_for_best_model \
    --group_by_length \
    --file_path $file_path \
    --load_in_4bit \
    --fp16 \
    --optim $optim
    