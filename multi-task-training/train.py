import numpy as np
import torch
import torch.nn as nn
import transformers
import logging
logging.basicConfig(level=logging.INFO)
from datasets import Dataset, DatasetDict
from typing import Any, Dict, List, NewType, Tuple, Optional, Union
from dataclasses import dataclass, field
from torch.nn.utils.rnn import pad_sequence
from transformers import HfArgumentParser, TrainingArguments
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import DataCollator, InputDataClass
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import os

tokenizer = None
model_args, data_args, training_args = None, None, None

os.environ["WANDB_DISABLED"] = "true"

@dataclass
class ModelArguments:
    # Base model parameters
    model_name_or_path: Optional[str] = field(default="xlm-roberta-large")
    mlm_probability: Optional[float] = field(default=0.3)

@dataclass
class DataArguments:
    
    pretrain_train_file_path: Optional[str] = field(default='data/en/wiki_auto/train_mt0.csv', metadata={"help": "Path to the training file."})
    pretrain_valid_file_path: Optional[str] = field(default='data/en/wiki_auto/valid_mt0.csv', metadata={"help": "Path to the training file."})

    sentiment_train_file_path: Optional[str] = field(default='data/en/wiki_auto/train_mt0.csv', metadata={"help": "Path to the training file."})
    sentiment_valid_file_path: Optional[str] = field(default='data/en/wiki_auto/valid_mt0.csv', metadata={"help": "Path to the training file."})
    sentiment_test_file_path: Optional[str] = field(default='data/en/wiki_auto/test_mt0.csv', metadata={"help": "Path to the training file."})

    max_length: Optional[int] = field(
        default=512, metadata={"help": "Maximum length of source. Sequences will be right padded (and possibly truncated)."}
    )


class NLPDataCollator(transformers.DefaultDataCollator):
    """
    Extending the existing DataCollator to work with NLP dataset batches
    """
    
    def collate_batch(self, features: List[Union[InputDataClass, Dict]]) -> Dict[str, torch.Tensor]:
        first = features[0]
        
        if isinstance(first, dict):
          # NLP data sets current works presents features as lists of dictionary
          # (one per example), so we  will adapt the collate_batch logic for that
          if "labels" in first and first["labels"] is not None:

              if first["labels"].dtype == torch.long:      
                  labels = torch.tensor([f["labels"].tolist() for f in features], dtype=torch.long)
              else:
                  labels = torch.tensor([f["labels"] for f in features], dtype=torch.float)
              
              batch = {"labels": labels}
          
          for k, v in first.items():
              if k != "labels" and v is not None and not isinstance(v, str):
                  batch[k] = torch.stack([f[k] for f in features])

          return batch
        
        else:
          # otherwise, revert to using the default collate_batch
          return transformers.DefaultDataCollator().collate_batch(features)


class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """
    def to(self, device):
        return self


class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """
    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)
    
    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """
    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        
        self.num_batches_dict = {
            task_name: len(dataloader) for task_name, dataloader in self.dataloader_dict.items()
        }
        
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(len(dataloader.dataset) for dataloader in self.dataloader_dict.values())

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """

        task_choice_list = []
        
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        
        dataloader_iter_dict = {
            task_name: iter(dataloader) for task_name, dataloader in self.dataloader_dict.items()
        }
        
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            #print("current task is ", task_name)
            yield next(dataloader_iter_dict[task_name])


class MultitaskTrainer(transformers.Trainer):

    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_sampler = (
            RandomSampler(train_dataset)
            # if self.args.local_rank == -1
            # else DistributedSampler(train_dataset)
        )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
              train_dataset,
              batch_size=self.args.train_batch_size,
              sampler=train_sampler,
              collate_fn=self.data_collator.collate_batch,
            ),
        )

        return data_loader

    def get_single_eval_dataloader(self, task_name, eval_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        
        if self.eval_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        eval_sampler = (
            SequentialSampler(eval_dataset)
            # if self.args.local_rank == -1
            # else DistributedSampler(eval_dataset)
        )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
              eval_dataset,
              batch_size=self.args.eval_batch_size,
              sampler=eval_sampler,
              collate_fn=self.data_collator.collate_batch,
            ),
        )

        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each 
        task Dataloader
        """
        return MultitaskDataloader({
            task_name: self.get_single_train_dataloader(task_name, task_dataset) for task_name, task_dataset in self.train_dataset.items()
        })

    
    def get_eval_dataloader(self, eval_dataset):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each 
        task Dataloader
        """

        # eval_dataloader = self.get_single_eval_dataloader("sentiment", eval_dataset)
        # print(eval_dataloader.data_loader)

        # return MultitaskDataloader({
        #     "sentiment": self.get_single_eval_dataloader("sentiment", eval_dataset)
        # })

        print(self.eval_dataset)
        print()
        exit()
        
        return self.get_single_eval_dataloader("sentiment", self.eval_dataset)

        # return MultitaskDataloader({
        #     task_name: self.get_single_eval_dataloader(task_name, task_dataset) for task_name, task_dataset in self.eval_dataset.items()
        # })


def _tensorize_batch(examples: List[torch.Tensor]) -> torch.Tensor:
    
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)
    else:
        if tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have one."
            )
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

def _tensorize_mask(examples: List[torch.Tensor]) -> torch.Tensor:
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)
    else:
        if tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have one."
            )
        return pad_sequence(examples, batch_first=True, padding_value=False)


def mask_tokens(inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )
    [inputs, tomask] = inputs
    labels = inputs.clone()
    
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, model_args.mlm_probability)
    
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]

    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    probability_matrix.masked_fill_(tomask.eq(False), value=0.0)
    
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    del tomask
    
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def convert_to_pretrain_features(example_batch):
    sentences = list(example_batch['sentence'])
    masks = list(example_batch['masks'])

    block_size = data_args.max_length

    tomasks = ['NOMASK {} NOMASK'.format(' '.join(mask.split()[:block_size - 2])).split() for mask in masks] 
    tomasks = [mask + ['NOMASK']*(block_size-len(mask)) for mask in tomasks if (block_size-len(mask)) > 0]
    tomasks = [[i == "MASK" for i in t] for t in tomasks]

    batch_encoding = tokenizer.batch_encode_plus(sentences, add_special_tokens=True, max_length=block_size, pad_to_max_length=True)

    for j in range(len(batch_encoding["input_ids"])):
        if len(batch_encoding["input_ids"][j]) != len(tomasks[j]):
            print(batch_encoding["input_ids"][j], sentences[j])
            print(f'{j}***{len(batch_encoding["input_ids"][j])}***{len(tomasks[j])}')
            break
    
    examples = batch_encoding["input_ids"] 

    examples = [torch.tensor(example, dtype=torch.long) for example in examples]
    tomasks = [torch.tensor(mask, dtype=torch.bool) for mask in tomasks]

    batch = _tensorize_batch(examples)
    tomasks = _tensorize_mask(tomasks)
    
    inputs, labels = mask_tokens([batch,tomasks])
    
    return {"input_ids": inputs, "labels": labels}


def convert_to_sentiment_features(example_batch):
    
    inputs = list(example_batch['sentence'])
    
    features = tokenizer.batch_encode_plus(
        inputs, max_length=data_args.max_length, add_special_tokens=True, padding="max_length"
    )

    label_dict = {"neutral": 0, "negative": 2, "positive": 1}
    labels = [torch.tensor(label_dict[label], dtype=torch.long) for label in example_batch["labels"]]

    features["labels"] = torch.stack(labels, dim=0)

    # print(features["labels"])
    # print()
    # print(features["input_ids"])
    # print()
    # print(features["labels"].size())
    # print(features["input_ids"].size())

    # exit()

    assert "input_ids" in features
    assert "attention_mask" in features

    return features



class MultitaskModel(transformers.PreTrainedModel):
    
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models. 

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name, 
                config=model_config_dict[task_name],
            )

            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            
            taskmodels_dict[task_name] = model
        
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        return "roberta"

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)



def main():
    
    global data_args
    global training_args
    global model_args

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # load pretraining dataset
    file = open(data_args.pretrain_train_file_path)
    pretrain_train_lines = file.readlines()[0:10]
    file.close()
    pretrain_train_lines_sentence = [line.split("\t")[0].strip().replace("\n", "") for line in pretrain_train_lines]
    pretrain_train_lines_mask = [line.split("\t")[1].strip().replace("\n", "") for line in pretrain_train_lines]
    pretrain_train = Dataset.from_dict({"sentence": pretrain_train_lines_sentence, "masks": pretrain_train_lines_mask})

    file = open(data_args.pretrain_valid_file_path)
    pretrain_valid_lines = file.readlines()[0:100]
    file.close()
    pretrain_valid_lines_sentence = [line.split("\t")[0].strip().replace("\n", "") for line in pretrain_valid_lines]
    pretrain_valid_lines_mask = [line.split("\t")[1].strip().replace("\n", "") for line in pretrain_valid_lines]
    pretrain_valid = Dataset.from_dict({"sentence": pretrain_valid_lines_sentence, "masks": pretrain_valid_lines_mask})

    pretrain_data = DatasetDict({"train": pretrain_train, "valid": pretrain_valid})

    # load bnsentiment classification dataset
    file = open(data_args.sentiment_train_file_path)
    sentiment_train_lines = file.readlines()
    file.close()
    sentiment_train_lines_sentence = [line.split("\t")[0].strip().replace("\n", "") for line in sentiment_train_lines]
    sentiment_train_lines_labels = [line.split("\t")[1].strip().replace("\n", "") for line in sentiment_train_lines]
    sentiment_train = Dataset.from_dict({"sentence": sentiment_train_lines_sentence, "labels": sentiment_train_lines_labels})

    file = open(data_args.sentiment_valid_file_path)
    sentiment_valid_lines = file.readlines()
    file.close()
    sentiment_valid_lines_sentence = [line.split("\t")[0].strip().replace("\n", "") for line in sentiment_valid_lines]
    sentiment_valid_lines_labels = [line.split("\t")[1].strip().replace("\n", "") for line in sentiment_valid_lines]
    sentiment_valid = Dataset.from_dict({"sentence": sentiment_valid_lines_sentence, "labels": sentiment_valid_lines_labels})

    file = open(data_args.sentiment_test_file_path)
    sentiment_test_lines = file.readlines()
    file.close()
    sentiment_test_lines_sentence = [line.split("\t")[0].strip().replace("\n", "") for line in sentiment_test_lines]
    sentiment_test_lines_labels = [line.split("\t")[1].strip().replace("\n", "") for line in sentiment_test_lines]
    sentiment_test = Dataset.from_dict({"sentence": sentiment_test_lines_sentence, "labels": sentiment_test_lines_labels})

    sentiment_data = DatasetDict({"train": sentiment_train, "valid": sentiment_valid, "test": sentiment_test})

    dataset_dict = {"pretrain": pretrain_data, "sentiment": sentiment_data}

    global tokenizer

    model_name = "xlm-roberta-large"
    
    multitask_model = MultitaskModel.create(
        model_name=model_name,
        model_type_dict={
            "pretrain": transformers.AutoModelForMaskedLM,
            "sentiment": transformers.AutoModelForSequenceClassification
        },
        
        model_config_dict={
            "pretrain": transformers.AutoConfig.from_pretrained(model_name),
            "sentiment": transformers.AutoConfig.from_pretrained(model_name, num_labels=3)
        },
    )

    print("Checking if encoder is same across")
    print()
    print(multitask_model.encoder.embeddings.word_embeddings.weight.data_ptr())
    print(multitask_model.taskmodels_dict["pretrain"].roberta.embeddings.word_embeddings.weight.data_ptr())
    print(multitask_model.taskmodels_dict["sentiment"].roberta.embeddings.word_embeddings.weight.data_ptr())
    print()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    convert_func_dict = {
        "pretrain": convert_to_pretrain_features,
        "sentiment": convert_to_sentiment_features
    }

    columns_dict = {
        "pretrain": ['input_ids', 'labels'],
        "sentiment": ['input_ids', 'attention_mask', 'labels']
    }

    features_dict = {}
    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        
        for phase, phase_dataset in dataset.items():
            
            features_dict[task_name][phase] = phase_dataset.map(
                convert_func_dict[task_name],
                batched=True,
                load_from_cache_file=False,
                batch_size=10
            )
            
            print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
            
            features_dict[task_name][phase].set_format(
                type="torch", 
                columns=columns_dict[task_name],
            )

            print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))

    train_dataset = {
        task_name: dataset["train"] 
        for task_name, dataset in features_dict.items()
    }
    # eval_dataset = {
    #     task_name: dataset["valid"] 
    #     for task_name, dataset in features_dict.items()
    # }

    eval_dataset = features_dict["sentiment"]["valid"]

    trainer = MultitaskTrainer(
        model=multitask_model,
        args=training_args,
        data_collator=NLPDataCollator(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    trainer.train()



if __name__ == "__main__":
    main()