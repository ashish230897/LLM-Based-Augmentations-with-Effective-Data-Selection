import pandas as pd
import os
import argparse

repo_path = os.getcwd()

bn_dict = {"positive": "ইতিবাচক", "negative": "নেতিবাচক", "neutral": "নিরপেক্ষ"}

def convert_text(data, labels):
    text_labels = []

    data_new = []
    for text in data:
        if text[-1] not in [".", "!", "?"]:
            data_new.append(text + ".")
        else:
            data_new.append(text)
    
    data = data_new
    
    i = 0
    for text, label in zip(data, labels):
        text_label = "{} \n{} </s>".format(label, text)
        text_labels.append(text_label)

        i += 1
    
    print("Lenght of texts is ", len(text_labels))

    dict = {"text_label": text_labels}
    
    return pd.DataFrame(dict)

def convert_text_bn(data, labels):
    text_labels = []

    data_new = []
    for text in data:
        if text[-1] not in [".", "!", "?"]:
            data_new.append(text + ".")
        else:
            data_new.append(text)
    
    data = data_new
    
    i = 0
    for text, label in zip(data, labels):
        text_label = "{} \n{} </s>".format(bn_dict[label], text)
        text_labels.append(text_label)

        i += 1
    
    print("Lenght of texts is ", len(text_labels))

    dict = {"text_label": text_labels}
    
    return pd.DataFrame(dict)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)

    args = parser.parse_args()

    data_path = args.input_path
    data, labels = [], []

    with open(data_path) as file:
        for line in file:
            data.append(line.split('\t')[0])
            labels.append(line.split('\t')[1].replace("\n", ""))

    print(len(data), len(labels))
    
    if args.lang == "bn":
        df = convert_text_bn(data, labels)
    else:
        df = convert_text(data, labels)

    print("First row is:")
    print(df.iloc[0])
    print("Length of dataframe is", len(df))

    df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()

