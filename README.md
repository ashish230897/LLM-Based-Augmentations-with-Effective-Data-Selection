# TaskSpecificGenerations

## Data
XNLI related data needs to be downloaded in the directory data/xnli/, which should be present in the home directory of this repository.  
Data related to pre-training should be downloaded in data/xnli/pretraining folder.  

## LLAMA-2 generations and finetuning scripts
Fine tuning code of llama-2 is present in the directory train_llama2. Also, the code to generate data for the different tasks using LLAMA-2 is present in this directory.

## Translation  
IndicTrans2 can be installed by following this: https://github.com/AI4Bharat/IndicTrans2#installation  
Clone this repository in the home directory and create a separate conda environment to translate using IndicTrans2.  
Download the en-indic model from https://indictrans2-public.objectstore.e2enetworks.net/it2_preprint_ckpts/en-indic-preprint.zip and store the weights in IndicTrans2/translations/en-indic-preprint folder.    





