import argparse
import pandas as pd
import re
import string

def remove_whitespace(text):
    return  " ".join(text.split())



def main(input_path, output_path):

    df = pd.read_csv(input_path)
    texts = list(df["Texts"])

    #text = "I'm not a Bengali, but I've heard a lot about you.  30 minutes of video, 15 minutes for the title, the rest for ads."
    texts = [''.join([i if ord(i) < 128 else '' for i in text]) for text in texts]
    texts = [remove_whitespace(text) for text in texts]
    texts = [re.sub(r'\d+\.\s+', '', text) for text in texts]  # remove full 2. in middle of texts
    
    
    punctuation_chars = re.escape(string.punctuation)
    pattern = r'\s*([' + punctuation_chars + r'])\s*'
    print("Pattern to remove spaces around punct is", pattern)
    texts = [re.sub(pattern, r'\1', text) for text in texts]  # remove spaces between punctuations

    punctuation_chars = re.escape(string.punctuation)
    pattern = r'([\d])([' + punctuation_chars + r'])\s*([^0-9])'
    print("Pattern to convert 90%hey to 90% hey", pattern)
    texts = [re.sub(pattern, r'\1\2 \3', text) for text in texts]

    punctuation_chars = re.escape(string.punctuation)
    pattern = r'([\w])([?.!,%])\s*([\w\d])'
    print("Pattern to add spaces after the above punctuations", pattern)
    texts = [re.sub(pattern, r'\1\2 \3', text) for text in texts]

    pattern = r'([' + punctuation_chars + r'])\s+\d+\.'
    print("Pattern to remove 2. at the end of strings is", pattern)
    texts = [re.sub(pattern, r'\1', text) for text in texts]  # remove 2. etc at the end of strings

    # pick texts having length > 4
    texts = [text for text in texts if len(text) > 4]

    texts = list(set(texts))

    # replace continuous punctuations with a single punctuation
    pattern = r'([^\w\s])\1+'
    texts = [re.sub(pattern, r'\1', text) for text in texts]

    # replace two dots with a single dot
    
    new_df = pd.DataFrame({"Texts": texts})
    new_df.to_csv(output_path, index=False)


def test(texts):

    texts = [''.join([i if ord(i) < 128 else ' ' for i in text]) for text in texts]
    texts = [remove_whitespace(text) for text in texts]
    texts = [re.sub(r'([^0-9])\d+\.\s+', r'\1.', text) for text in texts]  # remove full 2. in middle of texts
    print(texts[0])
    
    punctuation_chars = re.escape(string.punctuation)
    pattern = r'\s*([' + punctuation_chars + r'])\s*'
    print("Pattern to remove spaces around punct is", pattern)
    texts = [re.sub(pattern, r'\1', text) for text in texts]  # remove spaces between punctuations
    print(texts[0])

    punctuation_chars = re.escape(string.punctuation)
    pattern = r'([\d])([' + punctuation_chars + r'])\s*([^0-9])'
    print("Pattern to convert 90%hey to 90% hey", pattern)
    texts = [re.sub(pattern, r'\1\2 \3', text) for text in texts]
    print(texts[0])

    punctuation_chars = re.escape(string.punctuation)
    pattern = r'([\w])([?.!,%])\s*([\w\d])'
    print("Pattern to add spaces after the above punctuations", pattern)
    texts = [re.sub(pattern, r'\1\2 \3', text) for text in texts]

    print(texts[0])

    pattern = r'([' + punctuation_chars + r'])\s+\d+\.'
    print("Pattern to remove 2. at the end of strings is", pattern)
    texts = [re.sub(pattern, r'\1', text) for text in texts]  # remove 2. etc at the end of strings
    print(texts[0])




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=False, type=str)
    parser.add_argument("--output_path", required=False, type=str)

    args = parser.parse_args()
    

    # test(["It is better to show the faces of the people. Rana, Aparna, where is Bhaiya. 3."])
    main(args.input_path, args.output_path)