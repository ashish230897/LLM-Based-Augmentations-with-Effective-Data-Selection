import json
import random
import pandas as pd

# Opening JSON file
f = open('./data/diverse-CS.json')
 
data = json.load(f)
random.shuffle(data)

hi_inputs, cs_outputs = [], []

for dict in data:
    hi_inputs.append("code_mix: " + dict["hi_input"])
    cs_outputs.append(dict["cs_output"])

assert len(hi_inputs) == len(cs_outputs)


print(len(hi_inputs))

train_inputs, train_outputs = hi_inputs[0:4500], cs_outputs[0:4500]
valid_inputs, valid_outputs = hi_inputs[4500:], cs_outputs[4500:]

train_df = pd.DataFrame({"inputs": train_inputs, "outputs": train_outputs})
valid_df = pd.DataFrame({"inputs": valid_inputs, "outputs": valid_outputs})

train_df.to_csv("./data/train.csv", index=False)
valid_df.to_csv("./data/valid.csv", index=False)