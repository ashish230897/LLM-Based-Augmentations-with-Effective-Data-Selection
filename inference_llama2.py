from sklearn.model_selection import train_test_split
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, set_seed
import transformers
from peft import PeftConfig, PeftModel
import pandas as pd

saved_path = "./results/steps_120/final_model/"

config = PeftConfig.from_pretrained(saved_path)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    #bnb_4bit_compute_dtype=torch.float16,  # letting the compute data type be 32 bit
)

model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        use_auth_token="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"
    )
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True, use_auth_token="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")
tokenizer.pad_token = tokenizer.eos_token

# Load the LoRA model
inference_model = PeftModel.from_pretrained(model, saved_path)

def main():

    pipeline = transformers.pipeline(
        "text-generation",
        model=inference_model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )

    seeds = [i for i in range(100)]
    generations = []

    for i in seeds:

        # generate for different seeds
        print("Processing seed {}".format(i))
        set_seed(i)
        sequences = pipeline(
            ["<s>"],
            max_length=100,
            do_sample=True,
            top_k=20,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
        for seq in sequences:
            generations.append(seq[0]['generated_text'])
        
    
    generations_dict = {"generations": generations}
    df = pd.DataFrame(generations_dict)
    df.to_csv("./results/generations.csv", index=False)

if __name__ == "__main__":
    main()