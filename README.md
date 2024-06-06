# TaskSpecificGenerations  

This repo can be used to reproduce results of the paper "Boosting Zero-Shot Crosslingual Performance using LLM-Based Augmentations with Effective Data Selection"

## Data
Data related to all tasks need to stored in the data/ directory of the main repository. Datasets generated for all tasks are present here: https://drive.google.com/file/d/1t2aQAeKQMDK5GP8kg1oMtsZZDs11dpdz/view?usp=sharing  

## LLAMA-2 generations and finetuning scripts
Fine tuning code of llama-2 is present in the directory train_llama2. Also, the code to generate data for the different tasks using LLAMA-2 is present in this directory.

## Translation  
IndicTrans2 can be installed by following this: https://github.com/AI4Bharat/IndicTrans2#installation  
Clone this repository in the home directory and create a separate conda environment to translate using IndicTrans2.  
Download the en-indic model from https://indictrans2-public.objectstore.e2enetworks.net/it2_preprint_ckpts/en-indic-preprint.zip and store the weights in IndicTrans2/translations/en-indic-preprint folder.    


For any queries regarding the code base, reach out to: ashish.agrawal2123@gmail.com

If you use this repo, please cite the paper.


