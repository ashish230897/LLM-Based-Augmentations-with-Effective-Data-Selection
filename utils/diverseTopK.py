from nltk.cluster import KMeansClusterer
import nltk
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
from tqdm import tqdm
import random

model = SentenceTransformer('sentence-transformers/LaBSE', device='cuda')

def clustering_question(data, NUM_CLUSTERS = 15):

    X = np.array(data['emb'].tolist())
    print("shape of X is ", np.shape(X))

    kclusterer = KMeansClusterer(
        NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance,
        repeats=2, avoid_empty_clusters=True, rng=random.Random(42))

    assigned_clusters = kclusterer.cluster(X, assign_clusters=True, trace=True)

    data['cluster'] = pd.Series(assigned_clusters, index=data.index)
    data['centroid'] = data['cluster'].apply(lambda x: kclusterer.means()[x])

    return data, assigned_clusters


def distance_from_centroid(row):
    return nltk.cluster.util.cosine_distance(row['emb'], np.transpose(row['centroid']))


def cluster(input_path, class_):
    embeds_path = input_path + ".topk.{}.npy".format(class_)

    file = np.load(embeds_path, allow_pickle=True)
    texts, embeds, probs = [], [], []
    for ele in file:
        texts.append(ele[0])
        embeds.append(ele[1])
        probs.append(ele[2])
    
    df = pd.DataFrame({"Texts": texts, "emb": embeds, "probs": probs})
    data, _ = clustering_question(df, NUM_CLUSTERS=25)

    data['distance_from_centroid'] = data.apply(distance_from_centroid, axis=1)
    data.drop(["emb", "centroid"], axis=1, inplace=True)

    print(data.head())
    
    length = len(data)

    data.to_csv("{}.topk.{}.csv".format(input_path, class_), index=False)
    
    return length


def gather(input_path, class_, neu_length):

    df = pd.read_csv("{}.topk.{}.csv".format(input_path, class_))
    texts = list(df["Texts"])
    clusters = list(df["cluster"])
    probs = list(df["probs"])

    cluster_dict = {}
    for i in range(100):
        cluster_dict[i] = []
    
    for i in range(len(texts)):
        cluster_dict[clusters[i]].append([texts[i],probs[i]])

    total_available = 0
    for i in range(25):
        total_available += len(cluster_dict[i])

    # sort every list in descending order
    print(cluster_dict[0][0])
    for i in range(25):
        cluster_dict[i] = sorted(cluster_dict[i], key=lambda a: a[1], reverse=True)
    print(cluster_dict[0][0])

    total_cnt = 0
    new_texts = []

    if neu_length < 2500 and class_ == "pos":
        max_len = min(total_available, 2500 + (2500 - neu_length))
    else:
        max_len = min(total_available, 5000)


    curr_index = 0
    while total_cnt < max_len:
        for i in tqdm(range(25)):
            if total_cnt >= max_len: break

            if len(cluster_dict[i]) > curr_index:
                new_texts.append(cluster_dict[i][curr_index][0])
                total_cnt += 1

        curr_index += 1


    print("Total cnt is", total_cnt)
    df_new = pd.DataFrame({"Texts": new_texts})
    df_new.to_csv("{}.topk.{}_final.csv".format(input_path, class_), index=False)


def combine(input_path, train_path, out_path):
    pos_df = pd.read_csv("{}.topk.pos_final.csv".format(input_path))
    neu_df = pd.read_csv("{}.topk.neu_final.csv".format(input_path))
    neg_df = pd.read_csv("{}.topk.neg_final.csv".format(input_path))

    final_texts = list(neu_df["Texts"]) + list(pos_df["Texts"]) + list(neg_df["Texts"])
    final_texts = [text + "\t" + "neutral" + "\t" + "0" for text in final_texts]
    print(len(final_texts))

    file = open(train_path)
    lines = file.readlines()
    if len(lines[0].split("\t")) <= 2:
        lines = [line.split("\t")[0].strip() + "\t" + line.split("\t")[1].strip().replace("\n", "") + "\t" + "1" for line in lines]
    else: lines = [line.split("\t")[0].strip() + "\t" + line.split("\t")[1].strip().replace("\n", "") + "\t" + line.split("\t")[2].strip().replace("\n", "") + "\t" + "1" for line in lines]
    file.close()

    file = open(out_path, "w+")
    final = final_texts + lines
    for text in final:
        file.write(text + "\n")
    file.close()



def form_embeddings(input_path):
    neu, pos, neg = [], [], []

    file = open(input_path)
    lines = file.readlines()
    file.close()

    texts, embeds = [], []

    for line in lines:
        if len(line.split("\t")) <= 3:
            texts.append(line.split("\t")[0].strip())
        else:
            texts.append(line.split("\t")[0].strip() + "\t" + line.split("\t")[1].strip())
    
    print(texts[0])

    batch_size = 20
    num_batches = int(len(texts)/batch_size)

    for i in tqdm(range(num_batches)):
        embeddings = None
        if(i == num_batches - 1):
            curr_batch = texts[i*batch_size:]
            embeddings = model.encode(curr_batch)
        else:
            curr_batch = texts[i*batch_size:(i+1)*batch_size]
            embeddings = model.encode(curr_batch)
        
        embeds.extend(list(embeddings))
    
    print(np.shape(embeds))

    assert len(embeds) == len(lines)
    
    for line,embed in zip(lines, embeds):
        if len(line.split("\t")) > 3: label = line.split("\t")[2]
        else: label = line.split("\t")[1]
        if len(line.split("\t")) > 3: text = line.split("\t")[0].strip() + "\t" + line.split("\t")[1].strip()
        else: text = line.split("\t")[0].strip()
        
        # storing the probs too
        if len(line.split("\t")) > 3: prob = line.split("\t")[3]
        else: prob = line.split("\t")[2]
        
        if label == "neutral":
            neu.append([text, embed, float(prob)])
        elif label == "positive" or label == "entailment":
            pos.append([text, embed, float(prob)])
        elif label == "negative" or label == "contradiction":
            neg.append([text, embed, float(prob)])
        else:
            print("caughttt", text, label, float(prob))

    assert (len(neu) + len(pos) + len(neg)) == len(lines)

    np.save('{}.topk.neu.npy'.format(input_path), np.array(neu))
    np.save('{}.topk.neg.npy'.format(input_path), np.array(neg))
    np.save('{}.topk.pos.npy'.format(input_path), np.array(pos))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--class_", type=str)

    args = parser.parse_args()

    # if args.task == "encode":
    #     form_embeddings(args.input_path)
    # elif args.task == "cluster":
    #     cluster(args.input_path, args.class_)
    # elif args.task == "gather":
    #     gather(args.input_path, args.class_)
    # elif args.task == "combine":
    #     combine(args.input_path, args.train_path, args.out_path)
        
    
    # form_embeddings(args.input_path)
    # neu_length = cluster(args.input_path, "neu")
    # print("neutral length is ", neu_length)
    # cluster(args.input_path, "neg")
    # cluster(args.input_path, "pos")
    
    gather(args.input_path, "neu", 0)
    #gather(args.input_path, "pos", neu_length)
    gather(args.input_path, "pos", 2500)
    gather(args.input_path, "neg", 0)
    combine(args.input_path, args.train_path, args.out_path)

        

if __name__ == "__main__":
    main()