# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" SNLI utils (dataset loading and evaluation) """


import logging
import os

from transformers import DataProcessor
from .utils_custom import InputExample

logger = logging.getLogger(__name__)


class SnliProcessor_custom(DataProcessor):
    """Processor for the SNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass

    def get_examples(self, data_dir, language='en', split='train'):
      """See base class."""
      examples = []
      for lg in language.split(','):
        lines = self._read_tsv(os.path.join(data_dir, "{}-{}.tsv".format(split, lg)))
        print("!!!!!!!!!!!!!!!!! path is ", "{}-{}.tsv".format(split, lg))

        cnt = 0
        if split == "train":
            logger.info("\ntrain split in processor\n")
            for (i, line) in enumerate(lines):
                guid = "%s-%s-%s" % (split, lg, i)
                text_a = line[0]; text_b = line[1]
                
                label = str(line[2].strip().replace("\n", ""))
                # print("line no:{}\n".format(i))
                label_bool = int(line[3].strip().replace("\n", ""))  #1 for labeled data(0 for unlabeled data)
                
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=lg, label_bool = label_bool))
        else:
            for (i, line) in enumerate(lines):
                guid = "%s-%s-%s" % (split, lg, i)
                text_a = line[0]; text_b = line[1]
                
                if split == 'test' and len(line) != 3:
                    label = "neutral"
                else:
                    label = str(line[2].strip().replace("\n", ""))
                
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=lg))
        
      return examples
    #     for (i, line) in enumerate(lines):
    #       guid = "%s-%s-%s" % (split, lg, i)
    #       text_a = line[0]
    #       text_b = line[1]
          
    #       if cnt == 0:
    #         print(text_a, text_b)
    #         cnt += 1
          
    #       if split == 'test' and len(line) != 3:
    #         label = "neutral"
    #       else:
    #         label = str(line[2].strip().replace("\n", ""))
    #       assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
    #       #print(text_a, text_b, label)
    #       examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=lg))
    #   return examples

    def get_train_examples(self, data_dir, split, language='en'):
        return self.get_examples(data_dir, language, split)

    def get_dev_examples(self, data_dir, language='en'):
        return self.get_examples(data_dir, language, split='dev')

    def get_test_examples(self, data_dir, language='en'):
        return self.get_examples(data_dir, language, split='test')

    def get_translate_train_examples(self, data_dir, language='en'):
        """See base class."""
        examples = []
        for lg in language.split(','):
            file_path = os.path.join(data_dir, "SNLI-Translated/en-{}-translated.tsv".format(lg))
            logger.info("reading file from " + file_path)
            lines = self._read_tsv(file_path)
            for (i, line) in enumerate(lines):
                guid = "%s-%s-%s" % ("translate-train", lg, i)
                text_a = line[0]
                text_b = line[1]
                label = "contradiction" if line[2].strip() == "contradictory" else line[2].strip()
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=lg))
        return examples

    def get_translate_test_examples(self, data_dir, language='en'):
        lg = language
        lines = self._read_tsv(os.path.join(data_dir, "SNLI-Translated/test-{}-en-translated.tsv".format(lg)))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s-%s" % ("translate-test", language, i)
            text_a = line[0]
            text_b = line[1]
            label = "contradiction" if line[2].strip() == "contradictory" else line[2].strip()
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=language))
        return examples
        
    def get_pseudo_test_examples(self, data_dir, language='en'):
        lines = self._read_tsv(os.path.join(data_dir, "SNLI-Translated/pseudo-test-set/en-{}-pseudo-translated.csv".format(language)))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s-%s" % ("pseudo-test", language, i)
            text_a = line[0]
            text_b = line[1]
            label = "contradiction" if line[2].strip() == "contradictory" else line[2].strip()
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=language))
        return examples

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]


snli_processors = {
    "snli": SnliProcessor_custom,
}

snli_output_modes = {
    "snli": "classification",
}

snli_tasks_num_labels = {
    "snli": 3,
}
