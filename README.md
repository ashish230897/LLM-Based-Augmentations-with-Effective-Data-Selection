# TaskSpecificGenerations

## Data
XNLI related data needs to be downloaded in the directory data/xnli/, which should be present in the home directory of this repository.  
Data related to pre-training should be downloaded in data/xnli/pretraining folder.  

## Fine tuning llama-2
Fine tuning code of llama-2 is present in the directory train_llama2

## Translation  
IndicTrans2 can be installed by following this: https://github.com/AI4Bharat/IndicTrans2#installation  
Clone this repository in the home directory and create a separate conda environment to translate using IndicTrans2.  
Download the en-indic model from https://indictrans2-public.objectstore.e2enetworks.net/it2_preprint_ckpts/en-indic-preprint.zip and store the weights in IndicTrans2/translations/en-indic-preprint folder.  

## Results  
Create a directory "results" to store the trained models of llama-2 and xnli. Directories named "llama2" and "xnli" should be present in the results directory.  





