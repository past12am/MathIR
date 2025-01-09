# based on https://github.com/AnReu/ALBERT-for-Math-AR/blob/main/preprocessing_scripts/create_training_data_task1.py

import sys
import json
import random
from random import shuffle
from collections import defaultdict
from tqdm import tqdm
from os import makedirs
import pandas as pd
from sklearn.model_selection import train_test_split



def concatenate_answers(answers, corr_idx):
    res = ""
    start_char = None
    end_char = None
    for idx, answer in enumerate(answers):
        if (idx == corr_idx):
            start_char = len(res)

        res += "##{" + answer + "}##"

        if (idx == corr_idx):
            end_char = len(res)

    return res, start_char, end_char



def main(data_path, out_path, correct_percentage, min_answers, max_answers, filename_postfix, filename_postpostfix=""):
    collection_path = f'{data_path}/collection_{filename_postfix}.tsv'
    triples_path = f'{data_path}/triples_{filename_postfix}.jsonl'

    
    with open(collection_path, "r") as f:
        lines = f.readlines()

    collection_lines = list()
    for line in lines:
        collection_lines.append(line.split('\t', maxsplit=1))

    pd_collection = pd.DataFrame(collection_lines, columns=["pid", "doc"]) #pd.read_csv(collection_path, sep="\t", header=None, names=["pid", "doc"], on_bad_lines='warn')
    pd_collection["pid"] = pd_collection["pid"].astype(int)
    
    pd_triples = pd.read_json(path_or_buf=triples_path, lines=True)
    pd_triples.columns = ["qid", "pid+", "pid-"]

    makedirs(out_path, exist_ok=True)

    split_idx = int(correct_percentage * pd_triples.shape[0])
    #pd_corr, pd_dummy = train_test_split(pd_triples, test_size=correct_percentage)

    pd_corr = pd_triples.iloc[0:split_idx]
    pd_dummy = pd_triples.iloc[(split_idx+1)::]

    meta = list()
    collection_agg = list()

    print(f"Number of unique Queries: {pd_corr["qid"].nunique()}")

    prev_query_id = None
    ctr = 0
    for idx, corr_triple_row in tqdm(pd_corr.iterrows(), total=pd_corr.shape[0]):

        if(prev_query_id == int(corr_triple_row["qid"])):
            continue
        prev_query_id = int(corr_triple_row["qid"])

        num_wrong = random.randint(min_answers, max_answers)
        c_idx = random.randint(0, num_wrong - 1)

        # Get documents from other pds
        wrong_triples = pd_dummy.sample(num_wrong)

        all_samples = list()
        #for _, t in wrong_triples.iterrows():
            #wrong_sample = pd_collection[pd_collection.pid == t["pid+"]]
            #all_samples.append(wrong_sample["doc"].values[0])

        all_samples = wrong_triples["pid+"].values.tolist()

        #corr_sample = pd_collection[pd_collection.pid == corr_triple_row["pid+"]]["doc"].values[0]
        all_samples.insert(c_idx, int(corr_triple_row["pid+"]))

        #res, start_char, end_char = concatenate_answers(all_samples, c_idx)

        #meta.append({"qid": int(corr_triple_row["qid"]), "agg_id": ctr, "start_char": start_char, "end_char": end_char})
        collection_agg.append({"qid": int(corr_triple_row["qid"]), "agg_id": ctr, "pids": all_samples, "corr_idx": c_idx})

        ctr += 1

    
    with open(f'{out_path}/collection_agg_{filename_postfix}{filename_postpostfix}.json', 'w', encoding='utf-8') as f:
        f.write("[")

        for t_ctr, agg_sample in enumerate(collection_agg):
            f.write(json.dumps(agg_sample))

            if(t_ctr < len(collection_agg) - 1):
                f.write(",\n")
                
        f.write("]")

    #with open(f'{out_path}/meta_{filename_postfix}.json', 'w', encoding='utf-8') as f:
    #    f.write(json.dumps(meta))





if __name__ == "__main__":
    data_path = "./ARQMathAgg/dataset_v2/"
    out_path = "./ARQMathAgg/dataset_v2/aggregates/"
    correct_percentage = 0.5  # --> valid_p = 1 - train_p, no test set, because ARQMath provides test set

    min_answers = 24
    max_answers = 30

    main(data_path, out_path, correct_percentage, min_answers, max_answers, "test", "_extended")
