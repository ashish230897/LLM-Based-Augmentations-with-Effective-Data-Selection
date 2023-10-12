
repo=$PWD
data_dir="${repo}/data/bnsentiment/pretraining/"
model_type="xlmr"
saved_model_path="${repo}/results/sst5/ensst5finetunedv2/xlm-roberta-large-LR5e-6-epoch15-MaxLen128/checkpoint-best"
save_file_path="${repo}/data/bnsentiment/pretraining/zero-shot_topk_5.8k_each_en_ensst5ft.pl"
input_file="${repo}/data/bnsentiment/pretraining/pretrain_label_en.csv.pretrain.train.wl"

python $PWD/self-training/pseudo_label_as.py \
  --model_type $model_type \
  --data_dir $data_dir \
  --saved_model_path $saved_model_path \
  --save_file_path $save_file_path \
  --input_file $input_file