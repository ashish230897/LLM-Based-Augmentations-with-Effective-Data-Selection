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
import time
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

def process(sentence):

    sentence = [json.loads(sentence)["texta"], json.loads(sentence)["textb"]]
    inputs = tokenizer.encode_plus(sentence[0], sentence[1], add_special_tokens=True, max_length=256)
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    #print(input_ids)

    return {'input_ids': input_ids, "token_type_ids": token_type_ids}


def pseudo_label(args, l3cube_train_examples, is_fraction, is_random):
    
    print("Fraction bool is {}".format(is_fraction))
    print("Random bool is {}".format(is_random))
    print("Pseudo label all bool is {}".format(args.pl_all))
    # get label and its probability for each example
    global features
    global model

    features = []
    label_probs = []
    p = Pool(20)

    t1 = time.time()
    with p:
        features = p.map(process, l3cube_train_examples)
    print("Features length is {}".format(len(features)))
    print("Time taken is {}".format((time.time() - t1)/60))
    
    l3cube_train_examples = [[json.loads(ex)["texta"], json.loads(ex)["textb"]] for ex in l3cube_train_examples]

    # Evaluation
    model_ = torch.nn.DataParallel(model)
    # model_ = model
    print("device is", args.device)
    model_.to(args.device)

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
            model_.eval()
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0],
                            "attention_mask": batch[1], "labels": batch[2]} #, "token_type_ids": batch[3]}, xlm doesnt use segment ids
                outputs = model_(**inputs)#, return_dict=False)
                # print(outputs)
                # exit()
                prob = m(outputs[1])

                labels = list(torch.argmax(prob, dim=1).cpu().numpy())
                prob = prob.cpu().numpy()
                for j,index in enumerate(labels):
                    label_probs.append((index, prob[j][index]))
    
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
            model_.eval()
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0],
                            "attention_mask": batch[1],
                            "labels": batch[2]}
                outputs = model_(**inputs)#, return_dict=False)
                prob = m(outputs[1])
                labels = list(torch.argmax(prob, dim=1).cpu().numpy())
                prob = prob.cpu().numpy()
                for j,index in enumerate(labels):
                    label_probs.append((index, prob[j][index]))
    

    print("Finished pseudo labelling!")
    print("Length of label probs", len(label_probs))

    t1 = time.time()
    # collect neutral,positive,negative in separate lists and sort them
    negative = [("contradiction", tup[1], i) for i,(tup) in enumerate(label_probs) if tup[0] == 0]
    positive = [("entailment", tup[1], i) for i,(tup) in enumerate(label_probs) if tup[0] == 1]
    neutral = [("neutral", tup[1], i) for i,(tup) in enumerate(label_probs) if tup[0] == 2]

    print("Time taken 1: {}".format(time.time() - t1))

    print(len(neutral), len(positive), len(negative))

    t1 = time.time()

    if is_random:
        #shuffle the lists
        random.shuffle(neutral)
        random.shuffle(positive)
        random.shuffle(negative)
    elif not args.pl_all: # sort only if not pseudo label all
        neutral.sort(key=takeSecond, reverse=True)
        negative.sort(key=takeSecond, reverse=True)
        positive.sort(key=takeSecond, reverse=True)

    # save label and probabilities in a file
    # file = open(args.data_dir + "pseudolabeled/top_2.5k_each.pl", "w+")
    file = open(args.save_file_path, "w+")
    
    new_examples = []
    new_examples_dict = {}
    
    sents_per_class = args.sents_per_class
    # sents_per_class = 5800
    
    for _,prob,index in neutral:
        if l3cube_train_examples[index][0] + "\t" + l3cube_train_examples[index][1] not in new_examples_dict:
            if len(l3cube_train_examples[index][0].split()) >= 5 and len(l3cube_train_examples[index][1].split()) >= 5:
                new_examples.append({"texta": l3cube_train_examples[index][0], "textb": l3cube_train_examples[index][1], "label": "neutral", "prob": prob})
                new_examples_dict[l3cube_train_examples[index][0] + "\t" + l3cube_train_examples[index][1]] = 1
                if len(new_examples)  == sents_per_class and not args.pl_all:
                    break
    
    for _,prob,index in positive:
        if l3cube_train_examples[index][0] + "\t" + l3cube_train_examples[index][1] not in new_examples_dict:
            if len(l3cube_train_examples[index][0].split()) >= 5 and len(l3cube_train_examples[index][1].split()) >= 5:
                new_examples.append({"texta": l3cube_train_examples[index][0], "textb": l3cube_train_examples[index][1], "label": "entailment", "prob": prob})
                new_examples_dict[l3cube_train_examples[index][0] + "\t" + l3cube_train_examples[index][1]] = 1
                if len(new_examples)  == sents_per_class*2 and not args.pl_all:
                    break
    
    for _,prob,index in negative:
        if l3cube_train_examples[index][0] + "\t" + l3cube_train_examples[index][1] not in new_examples_dict:
            if len(l3cube_train_examples[index][0].split()) >= 5 and len(l3cube_train_examples[index][1].split()) >= 5:
                new_examples.append({"texta": l3cube_train_examples[index][0], "textb": l3cube_train_examples[index][1], "label": "contradiction", "prob": prob})
                new_examples_dict[l3cube_train_examples[index][0] + "\t" + l3cube_train_examples[index][1]] = 1
                if len(new_examples)  == sents_per_class*3 and not args.pl_all:
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
    parser.add_argument("--saved_model_path", default=None, type=str, required=False,
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
    parser.add_argument('--random', action='store_true')
    parser.add_argument("--save_file_path", type=str, default='')
    parser.add_argument("--input_file", type=str, default='')
    parser.add_argument('--pl_all', action='store_true')
    parser.add_argument("--sents_per_class", type=int, default=2500)

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
    
    l3cube_train_examples = read_examples_from_file(args.input_file)     
    l3cube_train_examples = [json.dumps({"texta": dict["texta"], "textb": dict["textb"]}) for dict in l3cube_train_examples]

    logger.info("Training/evaluation parameters %s", args)

    tokenizers = {"bert": BertTokenizer, "xlmr": XLMRobertaTokenizer}
    models = {"bert": BertForSequenceClassification, "xlmr": XLMRobertaForSequenceClassification}
    
    global tokenizer
    global model
    
    tokenizer = tokenizers[args.model_type].from_pretrained(args.saved_model_path, do_lower_case=True)
    model = models[args.model_type].from_pretrained(args.saved_model_path)
    model.to(args.device)

    pseudo_label(args, l3cube_train_examples, False, args.random)


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=3 python self-training/pseudo_label.py --model_type xlmr --eval_batch_size 512 
#export CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8

# /raid/speech/ashish/TSTG_new/results/sst5/hisst5finetunedv2/xlm-roberta-large-LR5e-6-epoch15-MaxLen128/checkpoint-best
# /raid/speech/ashish/TSTG_new/data/bnsentiment/pretraining/pretrain_label_en.csv.pretrain.train
# /raid/speech/ashish/TSTG_new/data/bnsentiment/pretraining/pretrain_label_en.csv.pretrain.train.wl