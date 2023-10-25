
repo=$PWD
data_dir="${repo}/data/marsentiment/pretraining/"
model_type="xlmr"
saved_model_path="${repo}/results/snli/hisnlifinetuned/xlm-roberta-large-LR5e-6-epoch15-MaxLen128/checkpoint-best"
save_file_path="${repo}/data/snli/pretraining/zero-shot_rand_2.5k_each_hi_hisnlift.pl"
input_file="${repo}/data/snli/pretraining/pretrain_label_hiprehypos.csv.pretrain.train.wl"

python $PWD/self-training/pseudo_label_snli.py \
  --model_type $model_type \
  --data_dir $data_dir \
  --saved_model_path $saved_model_path \
  --save_file_path $save_file_path \
  --input_file $input_file \
  --random