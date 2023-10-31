# Copyright (c) Microsoft Corporation. Licensed under the MIT license.
import argparse
import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Dataset
from tqdm import tqdm, trange
from transformers import (
    BertForSequenceClassification, BertTokenizer, XLMRobertaForSequenceClassification, XLMRobertaTokenizer, AdamW, get_linear_schedule_with_warmup
)
from sklearn.metrics import f1_score, precision_score, recall_score

from torch.nn import Softmax
from multiprocessing import Pool
import time,os
import json

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

logger = logging.getLogger(__name__)
features = []
tokenizer = None
args = None
model = None
device_ids = None


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    precision = precision_score(
        y_true=labels, y_pred=preds, average='weighted')
    recall = recall_score(y_true=labels, y_pred=preds, average='weighted')
    return{
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        "precision": precision,
        "recall": recall
    }


def read_examples_from_file(data_dir):
    file_path = data_dir

    examples = []
    with open(file_path, 'r') as infile:
        lines = infile.read().strip().split('\n')
    for line in lines:
        x = line.split('\t')
        texta = x[0]
        textb = x[1]
        label = x[2]
        examples.append({'texta': texta, 'textb': textb, 'label': label})
    
    return examples


class CustomDataset(Dataset):
    def __init__(self, input_ids, labels, token_type_ids, present=None):
        self.input_ids = input_ids
        self.labels = labels
        self.present = present
        self.token_type_ids = token_type_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.present:
            return torch.tensor(self.input_ids[i], dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.long), self.present[i]
        else:
            return torch.tensor(self.input_ids[i], dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.long), torch.tensor(self.token_type_ids[i], dtype=torch.long)


def collate(examples):
    padding_value = 0

    first_sentence = [t[0] for t in examples]
    first_sentence_padded = torch.nn.utils.rnn.pad_sequence(
        first_sentence, batch_first=True, padding_value=padding_value)

    max_length = first_sentence_padded.shape[1]
    first_sentence_attn_masks = torch.stack([torch.cat([torch.ones(len(t[0]), dtype=torch.long), torch.zeros(
        max_length - len(t[0]), dtype=torch.long)]) for t in examples])

    labels = torch.stack([t[1] for t in examples])
    # token_type_ids = torch.stack([t[2] for t in examples])
    token_type_ids = torch.stack([torch.cat([t[2], torch.zeros(
        max_length - len(t[2]), dtype=torch.long)]) for t in examples])

    # print(first_sentence_padded.size(), first_sentence_attn_masks.size(), labels.size(), token_type_ids.size())

    return first_sentence_padded, first_sentence_attn_masks, labels, token_type_ids

def takeSecond(ele):
    return ele[1]

def takeThird(ele):
    return ele[2]

def takeFourth(ele):
    return ele[3]

def process(sentence):

    sentence = [json.loads(sentence)["texta"], json.loads(sentence)["textb"]]
    inputs = tokenizer.encode_plus(sentence[0], sentence[1], add_special_tokens=True, max_length=256)
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    #print(input_ids)

    return {'input_ids': input_ids, "token_type_ids": token_type_ids}


def pseudo_label(args, models, train_examples, is_fraction):
    
    print("Fraction bool is {}".format(is_fraction))
    # print("Random bool is {}".format(is_random))
    # get label and its probability for each example
    global features
    global model

    features = []
    label_probs = []
    p = Pool(20)

    t1 = time.time()
    with p:
        features = p.map(process, train_examples)
    print("Features length is {}".format(len(features)))
    print("Time taken is {}".format((time.time() - t1)/60))
    
    train_examples = [[json.loads(ex)["texta"], json.loads(ex)["textb"]] for ex in train_examples]

    # # Evaluation
    # model_ = torch.nn.DataParallel(model)
    # # model_ = model
    # print("device is", args.device)
    # model_.to(args.device)

    #create a list of size (no of instances, no. of labels(3), no. of model checkpoints) to store the predicted probabilities
    probs_cart = {}
    # form batches of size 10,00,000
    # Convert to Tensors and build dataset
    batch_size = 128
    num_batches = int(len(features)/batch_size)
    print("Number of batches are {}".format(num_batches))
    m = Softmax(dim=-1)
    for batch in tqdm(range(num_batches)):
        current_batch = features[batch*batch_size: batch*batch_size+batch_size]
        all_input_ids = [f['input_ids'] for f in current_batch]
        all_token_type_ids = [f['token_type_ids'] for f in current_batch]
        all_labels = [0 for _ in current_batch]
        args_ = [all_input_ids, all_labels, all_token_type_ids]
        dataset = CustomDataset(*args_)
        
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(
            dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate, num_workers=2)
        
        with torch.no_grad():
            #evaluate using each of the model checkpoints
            for ckpt in range(args.num_checkpoints):    
                # model_path = os.path.join(args.saved_model_path, "checkpoint-{}".format(ckpt))
                # model = models[args.model_type].from_pretrained(model_path)
                # model_ = torch.nn.DataParallel(model)
                # # print("device is", args.device)
                # model_.to(args.device)
                # model_.eval()
                model = models[ckpt]
                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    batch = tuple(t.to(args.device) for t in batch)
                    inputs = {"input_ids": batch[0],
                                "attention_mask": batch[1], "labels": batch[2]} #, "token_type_ids": batch[3]}, xlm doesnt use segment ids
                    outputs = model(**inputs)#, return_dict=False)
                    # print(outputs)
                    # exit()
                    prob = m(outputs[1])

                    labels = list(torch.argmax(prob, dim=1).cpu().numpy())
                    prob = prob.cpu().numpy()
                    if ckpt == args.best_checkpoint_index - 1: #best_checkpoint_index:
                        for j,index in enumerate(labels):
                            label_probs.append((index, prob[j][index]))
                    print("Shape of probs:",prob.shape)
                    #record the predicted probabilites of each label for each instance in the batch
                    if ckpt not in probs_cart.keys():
                        probs_cart[ckpt] = [[prob[j][0],prob[j][1],prob[j][2]] for j in range(prob.shape[0])]
                    else:
                        probs_cart[ckpt].extend([prob[j][0],prob[j][1],prob[j][2]] for j in range(prob.shape[0]))


    
    if num_batches*batch_size < len(features):
        current_batch = features[num_batches*batch_size:]
        all_input_ids = [f['input_ids'] for f in current_batch]
        all_token_type_ids = [f['token_type_ids'] for f in current_batch]
        all_labels = [0 for _ in current_batch]
        args_ = [all_input_ids, all_labels, all_token_type_ids]
        dataset = CustomDataset(*args_)
        
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(
            dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate, num_workers=2)
        
        with torch.no_grad():
            for ckpt in range(args.num_checkpoints):     
                # model_path = os.path.join(args.?, "checkpoint-{}".format(ckpt))
                # model = models[args.model_type].from_pretrained(model_path)
                # model_ = torch.nn.DataParallel(model)
                # # print("device is", args.device)
                # model_.to(args.device)
                # model_.eval()
                model = models[ckpt]
                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    batch = tuple(t.to(args.device) for t in batch)
                    inputs = {"input_ids": batch[0],
                                "attention_mask": batch[1],
                                "labels": batch[2]}
                    outputs = model(**inputs)#, return_dict=False)
                    prob = m(outputs[1])
                    labels = list(torch.argmax(prob, dim=1).cpu().numpy())
                    prob = prob.cpu().numpy()
                    print("Shape of probs later:",prob.shape)
                    if ckpt == args.best_checkpoint_index - 1 : #best_checkpoint_index:
                        for j,index in enumerate(labels):
                            label_probs.append((index, prob[j][index]))
                    #record the predicted probabilites of each label
                    # if ckpt in probs_cart.keys():
                    #     probs_cart[ckpt] = [[prob[j][0],prob[j][1],prob[j][2]] for j in len(labels)]
                    # else:
                    probs_cart[ckpt].extend([prob[j][0],prob[j][1],prob[j][2]] for j in range(prob.shape[0]))
    print("Size of prob_cart: ({},{},{})".format(len(probs_cart),len(probs_cart[0]),len(probs_cart[0][0])))
    #label_probs should have the labels based on the last checkpoint (change this to best later based on eval scores on snli)
    print("Finished pseudolabeling(using best checkpoint) and gathering probabilities!")
    print("Length of label probs", len(label_probs))
    #dictionary to list:
    probs_cart_list = [value for key, value in sorted(probs_cart.items())]
    #compute variability scores::
    probs_cart_np =  np.array(probs_cart_list).transpose(1, 2, 0) #N,L*C -> N,L,C
    std_dev = np.std(probs_cart_np, axis=2)  # -> (N,L,)
    print(std_dev.shape)
    estimated_max_variability = np.max(std_dev, axis=-1).reshape(-1, 1)  # -> 
    print(estimated_max_variability.shape)
    #compute confidence scores::
    mean = np.average(probs_cart_np, axis=2) 
    print(mean.shape)
    estimated_max_confidence = np.max(mean, axis=-1).reshape(-1, 1)
    print(estimated_max_confidence.shape)

    t1 = time.time()
    # collect neutral,positive,negative in separate lists and sort them
    negative = [("contradiction", tup[1], estimated_max_variability[i],estimated_max_confidence[i],i) for i,(tup) in enumerate(label_probs) if tup[0] == 0]
    positive = [("entailment", tup[1],estimated_max_variability[i],estimated_max_confidence[i], i) for i,(tup) in enumerate(label_probs) if tup[0] == 1]
    neutral = [("neutral", tup[1],estimated_max_variability[i],estimated_max_confidence[i], i) for i,(tup) in enumerate(label_probs) if tup[0] == 2]

    print("Time taken 1: {}".format(time.time() - t1))

    print(len(neutral), len(positive), len(negative))

    t1 = time.time()

    if args.ambiguity == "ambiguous":
        neutral.sort(key=takeThird, reverse=True)
        negative.sort(key=takeThird, reverse=True)
        positive.sort(key=takeThird, reverse=True)
    else: #ambiguity == "easy":
        neutral.sort(key=takeFourth, reverse=True)
        negative.sort(key=takeFourth, reverse=True)
        positive.sort(key=takeFourth, reverse=True)

    # save label and probabilities in a file
    # file = open(args.data_dir + "pseudolabeled/top_2.5k_each.pl", "w+")
    file = open(args.save_file_path, "w+")
    
    new_examples = []
    new_examples_dict = {}
    
    # args.sents_per_class = 2500
    # sents_per_class = 5800
    
    for _,prob,_,_,index in neutral:
        if train_examples[index][0] + "\t" + train_examples[index][1] not in new_examples_dict:
            if len(train_examples[index][0].split()) >= 5 and len(train_examples[index][1].split()) >= 5:
                new_examples.append({"texta": train_examples[index][0], "textb": train_examples[index][1], "label": "neutral", "prob": prob})
                new_examples_dict[train_examples[index][0] + "\t" + train_examples[index][1]] = 1
                if len(new_examples)  == args.sents_per_class:
                    break
    
    for _,prob,_,_,index in positive:
        if train_examples[index][0] + "\t" + train_examples[index][1] not in new_examples_dict:
            if len(train_examples[index][0].split()) >= 5 and len(train_examples[index][1].split()) >= 5:
                new_examples.append({"texta": train_examples[index][0], "textb": train_examples[index][1], "label": "entailment", "prob": prob})
                new_examples_dict[train_examples[index][0] + "\t" + train_examples[index][1]] = 1
                if len(new_examples)  == args.sents_per_class*2:
                    break
    
    for _,prob,_,_,index in negative:
        if train_examples[index][0] + "\t" + train_examples[index][1] not in new_examples_dict:
            if len(train_examples[index][0].split()) >= 5 and len(train_examples[index][1].split()) >= 5:
                new_examples.append({"texta": train_examples[index][0], "textb": train_examples[index][1], "label": "contradiction", "prob": prob})
                new_examples_dict[train_examples[index][0] + "\t" + train_examples[index][1]] = 1
                if len(new_examples)  == args.sents_per_class*3:
                    break

    print("Time taken 2: {}".format(time.time() - t1))

    for example in new_examples:
        file.write(example["texta"] + "\t" + example["textb"] + "\t" + example["label"] + "\t" + str(example["prob"]) + "\n")
    file.close()


def main():

    global args

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=False,
                        help="The input data dir")
    parser.add_argument("--saved_model_path", default=None, type=str, required=True,
                        help="The saved model path")

    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # Optional Parameters
    parser.add_argument("--learning_rate", default=1.5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--device_index", default=0, type=int,
                        help="The cuda device on which the model will train.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=1024, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--model_type", type=str,
                        default='bert', help='type of model xlm/xlmr/bert')
    parser.add_argument("--model_name", default='bert-base-multilingual-cased',
                        type=str, help='name of pretrained model/path to checkpoint')
    parser.add_argument("--save_steps", type=int, default=1, help='set to -1 to not save model')
    parser.add_argument("--max_seq_length", default=256, type=int, help="max seq length after tokenization")
    parser.add_argument("--local_rank", default=0, type=int)
    # parser.add_argument('--random', action='store_true')
    parser.add_argument("--ambiguity", type=str,
                        default='ambiguous', help='ambiguous/easy')
    parser.add_argument("--save_file_path", type=str, default='')
    parser.add_argument("--input_file", type=str, default='',required=True)
    parser.add_argument("--num_checkpoints", default=15, type=int, help="Number of checkpoints saved for the trained model")
    parser.add_argument("--best_checkpoint_index", default=15, type=int, help="Checkpoint index to be used for deriving (pseudo)labels")
    parser.add_argument("--sents_per_class", default=2500, type=int, help="How many instances should be selected for each class")

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")
    args.device = device
    print(args.device)

    # Set up logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)

    # Set seed
    set_seed(args)
    
    train_examples = read_examples_from_file(args.input_file)     
    train_examples = [json.dumps({"texta": dict["texta"], "textb": dict["textb"]}) for dict in train_examples]

    logger.info("Training/evaluation parameters %s", args)

    tokenizers = {"bert": BertTokenizer, "xlmr": XLMRobertaTokenizer}
    models = {"bert": BertForSequenceClassification, "xlmr": XLMRobertaForSequenceClassification}
    
    global tokenizer
    global model
    
    first_model_path = os.path.join(args.saved_model_path, "checkpoint-0")
    tokenizer = tokenizers[args.model_type].from_pretrained(first_model_path, do_lower_case=True)
    # model = models[args.model_type].from_pretrained(args.saved_model_path)
    # model.to(args.device)
    model_ckpts = []
    for ckpt in range(args.num_checkpoints):    
        model_path = os.path.join(args.saved_model_path, "checkpoint-{}".format(ckpt))
        model = models[args.model_type].from_pretrained(model_path)
        model_ = torch.nn.DataParallel(model)
        model_.to(args.device)
        model_.eval()
        model_ckpts.append(model_)

    pseudo_label(args, model_ckpts, train_examples, False)


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=4 nohup python self-training/pseudo_label_xnli_cartography.py --model_type xlmr --eval_batch_size 512 --input_file data/snli/pretraining/pretrain_label_enprehypos.csv.pretrain.train.wl --saved_model_path /raid/speech/ashish/TSTG_new/results/xnli/en-ckpt-output  --save_file_path /raid/speech/ashish/TSTG_new/data/snli/zero-shot_ambiguous/expt-3/ambiguous-en.txt --num_checkpoints 15 --best_checkpoint_index 15   > logs-xnli/carto-en-ambiguous &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/pseudo_label_xnli_cartography.py --model_type xlmr --eval_batch_size 512 --input_file data/snli/pretraining/pretrain_label_hiprehypos.csv.pretrain.train.wl --saved_model_path /raid/speech/ashish/TSTG_new/results/xnli/hi-ckpt-output  --save_file_path /raid/speech/ashish/TSTG_new/data/snli/zero-shot_ambiguous/expt-2/ambiguous-hi.txt --num_checkpoints 15 --best_checkpoint_index 15  > logs-xnli/carto-hi-ambiguous &
# CUDA_VISIBLE_DEVICES=7 nohup python self-training/pseudo_label_xnli_cartography.py --model_type xlmr --eval_batch_size 512 --input_file data/snli/pretraining/pretrain_label_urprehypos.csv.pretrain.train.wl --saved_model_path /raid/speech/ashish/TSTG_new/results/xnli/ur-ckpt-output  --save_file_path /raid/speech/ashish/TSTG_new/data/snli/zero-shot_ambiguous/expt-2/ambiguous-ur.txt --num_checkpoints 15 --best_checkpoint_index 15   > logs-xnli/carto-ur-ambiguous &
# CUDA_VISIBLE_DEVICES=7 nohup python self-training/pseudo_label_xnli_cartography.py --model_type xlmr --eval_batch_size 512 --input_file data/snli/pretraining/pretrain_label_swprehypos.csv.pretrain.train.wl --saved_model_path /raid/speech/ashish/TSTG_new/results/xnli/sw-ckpt-output  --save_file_path /raid/speech/ashish/TSTG_new/data/snli/zero-shot_ambiguous/expt-2/ambiguous-sw.txt --num_checkpoints 15 --best_checkpoint_index 15   > logs-xnli/carto-sw-ambiguous &

# CUDA_VISIBLE_DEVICES=3 nohup python self-training/pseudo_label_xnli_cartography.py --model_type xlmr --eval_batch_size 512 --input_file data/snli/pretraining/pretrain_label_enprehypos.csv.pretrain.train.wl --saved_model_path /raid/speech/ashish/TSTG_new/results/xnli/en-ckpt-output  --save_file_path /raid/speech/ashish/TSTG_new/data/snli/zero-shot_easy/expt-3/easy-en.txt --num_checkpoints 15 --best_checkpoint_index 15 --ambiguity easy  > logs-xnli/carto-en-easy &
# CUDA_VISIBLE_DEVICES=3 nohup python self-training/pseudo_label_xnli_cartography.py --model_type xlmr --eval_batch_size 512 --input_file data/snli/pretraining/pretrain_label_hiprehypos.csv.pretrain.train.wl --saved_model_path /raid/speech/ashish/TSTG_new/results/xnli/hi-ckpt-output  --save_file_path /raid/speech/ashish/TSTG_new/data/snli/zero-shot_easy/expt-2/easy-hi.txt --num_checkpoints 15 --best_checkpoint_index 15 --ambiguity easy > logs-xnli/carto-hi-easy &
# CUDA_VISIBLE_DEVICES=6 nohup python self-training/pseudo_label_xnli_cartography.py --model_type xlmr --eval_batch_size 512 --input_file data/snli/pretraining/pretrain_label_urprehypos.csv.pretrain.train.wl --saved_model_path /raid/speech/ashish/TSTG_new/results/xnli/ur-ckpt-output  --save_file_path /raid/speech/ashish/TSTG_new/data/snli/zero-shot_easy/expt-2/easy-ur.txt --num_checkpoints 15 --best_checkpoint_index 15  --ambiguity easy > logs-xnli/carto-ur-easy &
# CUDA_VISIBLE_DEVICES=6 nohup python self-training/pseudo_label_xnli_cartography.py --model_type xlmr --eval_batch_size 512 --input_file data/snli/pretraining/pretrain_label_swprehypos.csv.pretrain.train.wl --saved_model_path /raid/speech/ashish/TSTG_new/results/xnli/sw-ckpt-output  --save_file_path /raid/speech/ashish/TSTG_new/data/snli/zero-shot_easy/expt-2/easy-sw.txt --num_checkpoints 15 --best_checkpoint_index 15  --ambiguity easy > logs-xnli/carto-sw-easy &

#export CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8

# /raid/speech/ashish/TSTG_new/results/sst5/hisst5finetunedv2/xlm-roberta-large-LR5e-6-epoch15-MaxLen128/checkpoint-best
# /raid/speech/ashish/TSTG_new/data/bnsentiment/pretraining/pretrain_label_en.csv.pretrain.train
# /raid/speech/ashish/TSTG_new/data/bnsentiment/pretraining/pretrain_label_en.csv.pretrain.train.wl

# cat file1.txt file2.txt | shuf > shuffled_file.txt
# cat data/snli/zero-shot_ambiguous/expt-2/ambiguous-sw.bl data/snli/train-sw-bl.tsv | shuf > data/snli/zero-shot_ambiguous/expt-2/train-sw.tsv