from datasets import concatenate_datasets
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from datasets import Dataset

model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_auth_token="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")

train_df = pd.read_csv("/raid/speech/ashish/TSTG_new/data/bnsentiment/train-en_texts.csv")

train = Dataset.from_pandas(train_df)

# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = train.map(lambda x: tokenizer(x["text_label"], truncation=True), batched=True, remove_columns=["text_label"])
input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]

# take 85 percentile of max length for better utilization
max_source_length = int(np.percentile(input_lenghts, 85))
print(f"Max source length: {max_source_length}")