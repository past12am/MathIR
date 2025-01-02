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

def select_rand_incorrect_answer(corr_entry, questions_with_answers, rand_other_tag_override=None):
    rand_other_tag = random.choice(corr_entry['tags']) if rand_other_tag_override is None else rand_other_tag_override

    if len(questions_with_answers[rand_other_tag]) == 1:
        for i in range(len(corr_entry['tags'])):
            rand_other_tag = corr_entry['tags'][i]

            if len(questions_with_answers[rand_other_tag]) > 1:
                break

        rand_other_tag = None


    if(rand_other_tag is None or len(questions_with_answers[rand_other_tag]) <= 1):     # TODO configurable Thresh
        rand_other_tag = random.choice(list(questions_with_answers.keys()))

    # at this point we have a tag that will definitely yield a result

    rand_other_entry = None

    while (rand_other_entry is None):
        rand_other_entry = random.choice(questions_with_answers[rand_other_tag])

        if (rand_other_entry["post_id"] == corr_entry["post_id"]):
            rand_other_entry = None

    return rand_other_entry



def concatenate_answers(answers, corr_idx=None):
    res = ""
    start_char = None
    end_char = None
    for idx, answer in enumerate(answers):
        if (idx == corr_idx):
            start_char = len(res)

        res += "{" + answer + "}"

        if (idx == corr_idx):
            end_char = len(res)

    return res, start_char, end_char


def create_dataset(data, out_path, filename_postfix):

    # 1. Remove questions without answers
    # 2. Group questions by tag
    questions_with_answers = defaultdict(list)
    all_answer_ids = set()
    for q in data:
        if 'answers' not in q:
            continue  # we only want questions with answers

        for tag in q['tags']:
            questions_with_answers[tag].append(q)

        for pid in q['answer_ids']:
            all_answer_ids.add(pid)



    # 3. Check number of questions for each tag
    print('Questions with answers, sizes by tag:')
    for tag in questions_with_answers:
        print(tag, len(questions_with_answers[tag]))



    # 4. For each questions: get one correct answer (random out of all answers of this question) and one incorrect answer with at least one common tag
    queries = list()
    collection = list()
    triples = list()
    meta = list()
    qrel = list()


    # Relabel answers to be in increasing order
    answer_ctr = 0
    for entry in tqdm(data):
        if 'answers' not in entry:
            continue

        for i in range(len(entry["answer_ids"])):
            entry["answer_ids"][i] = answer_ctr + i
        
        answer_ctr += len(entry["answer_ids"])


    for d in tqdm(data):
        question = d["question"]

        correct_answer = None

        # Sample Correct
        if 'answers' in d:
            correct_answer = random.choice(d['answers'])
        else:
            continue

        
        # Build Triples
        triples.append({"qid": d["post_id"], "pid+": list(), "pid-": list()})
        for correct_answer, correct_answer_id in zip (d['answers'], d["answer_ids"]):

            # Sample Incorrect
            num_wrong = 1
            
            wrong_answer_entry = None
            fail_ctr = 0
            while (wrong_answer_entry is None):

                if(fail_ctr <= 2 * num_wrong):
                    wrong_answer_entry = select_rand_incorrect_answer(d, questions_with_answers)
                else:
                    # we need to sample differently
                    wrong_answer_entry = select_rand_incorrect_answer(d, questions_with_answers, random.choice(list(questions_with_answers.keys())))


            triples[-1]["pid+"].append(correct_answer_id)
            triples[-1]["pid-"].append(random.choice(wrong_answer_entry["answer_ids"]))
            


        #Put together dataset
        queries.append({"qid": d["post_id"], "query": question})
        
        # Store all answers with their id, (the incorrect can be skipped, as they are the correct for another question)
        for correct_answer, correct_answer_id in zip (d['answers'], d["answer_ids"]):
            collection.append({"pid": correct_answer_id, "doc": correct_answer})

            qrel.append({"qid": d["post_id"], "pid": correct_answer_id, "relevance": 1})


    # Save dataset to files

    with open(f'{out_path}/queries_{filename_postfix}.tsv', 'w', encoding='utf-8') as f:
        for query in queries:
            f.write(f"{query['qid']}\t{query['query']}\n")

    with open(f'{out_path}/collection_{filename_postfix}.tsv', 'w', encoding='utf-8') as f:
        for doc in collection:
            f.write(f"{doc['pid']}\t{doc['doc']}\n")

    with open(f'{out_path}/triples_{filename_postfix}.jsonl', 'w', encoding='utf-8') as f:
        for t in triples:
            for pos_id, neg_id in zip(t["pid+"], t["pid-"]):
                f.write(f"[{t['qid']},{pos_id},{neg_id}]\n")
    

    with open(f'{out_path}/qrel_{filename_postfix}', 'w', encoding='utf-8') as f:
        for qrel_entry in qrel:
            f.write(f"{qrel_entry["qid"]} 0 {qrel_entry["pid"]} {qrel_entry["relevance"]}\n")


def main(data_path, out_path, train_p):
    data = json.load(open(f'{data_path}/cleaned_with_links.json', encoding='utf-8'))
    makedirs(out_path, exist_ok=True)

    # Split train and test parts
    train, test = train_test_split(data, train_size=train_p)
    create_dataset(train, out_path, "train")
    create_dataset(test, out_path, "test")



if __name__ == "__main__":
    data_path = "../ARQMath/data_preprocessing/"
    out_path = "../ARQMathAgg/dataset_v2/"
    train_p = 0.7  # --> valid_p = 1 - train_p, no test set, because ARQMath provides test set

    main(data_path, out_path, train_p)
