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


def create_dataset(data, out_path, filename_postfix, min_answers, max_answers):

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

    doc_ctr = 0

    doc_maxlen = 0
    doc_avglen = 0
    query_maxlen = 0
    query_avglen = 0
    samples_ctr = 0

    for d in tqdm(data):
        question = d["question"]

        correct_answer = None
        all_correct_answers_ids = None

        # Sample Correct
        if 'answers' in d:
            correct_answer = random.choice(d['answers'])
            all_correct_answers_ids = d["answer_ids"]
            #correct_pairs.append((d['title'] + ' ' + d['question'] , correct_answer, '1')) # Label 1 for correct question-answer pairs
        else:
            continue

        # Sample Incorrect
        num_wrong = random.randint(min_answers, max_answers)
        
        wrong_answer_post_ids = list()
        wrong_answer_entries = list()
        fail_ctr = 0
        while (len(wrong_answer_entries) < num_wrong):

            if(fail_ctr <= 2 * num_wrong):
                wrong_answer_entry = select_rand_incorrect_answer(d, questions_with_answers)
            else:
                # we need to sample differently
                wrong_answer_entry = select_rand_incorrect_answer(d, questions_with_answers, random.choice(list(questions_with_answers.keys())))

            
            if(wrong_answer_entry["post_id"] in wrong_answer_post_ids):
                fail_ctr += 1
                continue

            wrong_answer_entries.append(wrong_answer_entry)
            wrong_answer_post_ids.append(wrong_answer_entry["post_id"])


        # Build Correct and incorrect documents
        wrong_answers = [ answer for wrong_answer_entry in wrong_answer_entries for answer in wrong_answer_entry["answers"] ]
        wrong_answer_ids = [ answer for wrong_answer_entry in wrong_answer_entries for answer in wrong_answer_entry["answer_ids"] ]

        corr_fraction = random.randint(int(0.2 * len(wrong_answers)), int(0.8 * len(wrong_answers)))

        corr_idx = random.randint(0, corr_fraction - 1) if corr_fraction > 1 else 0
        corr_doc_list = wrong_answers[0:corr_fraction]
        corr_doc_list.insert(corr_idx, correct_answer)
        

        corr_doc, start_char, end_char = concatenate_answers(corr_doc_list, corr_idx)

        false_doc, _, _ = concatenate_answers(wrong_answers[corr_fraction::])


        #Put together dataset
        queries.append({"qid": d["post_id"], "query": question})

        query_len = len(question.split(' '))
        query_avglen += query_len
        if(query_len >= query_maxlen):
            query_maxlen = query_len
        
        collection.append({"pid": doc_ctr, "doc": corr_doc})
        collection.append({"pid": doc_ctr + 1, "doc": false_doc})


        doc_pos_len = len(corr_doc.split(' '))
        doc_neg_len = len(false_doc.split(' '))
        doc_avglen += doc_pos_len
        doc_avglen += doc_neg_len

        if(doc_pos_len >= doc_maxlen):
            doc_maxlen = doc_pos_len
        if(doc_neg_len >= doc_maxlen):
            doc_maxlen = doc_neg_len

        triples.append({"qid": d["post_id"], "pid+": doc_ctr, "pid-": doc_ctr + 1})

        meta.append({"qid": d["post_id"], "pid+": doc_ctr, "pid-": doc_ctr + 1, "start_char": start_char, "end_char": end_char})

        for aid in all_correct_answers_ids:
            qrel.append({"qid": d["post_id"], "pid": aid, "relevance": 1})

        doc_ctr += 2
        samples_ctr += 1


    # Save dataset to files

    with open(f'{out_path}/queries_{filename_postfix}.tsv', 'w', encoding='utf-8') as f:
        for query in queries:
            f.write(f"{query['qid']}\t{query['query']}\n")

    with open(f'{out_path}/collection_{filename_postfix}.tsv', 'w', encoding='utf-8') as f:
        for doc in collection:
            f.write(f"{doc['pid']}\t{doc['doc']}\n")

    with open(f'{out_path}/triples_{filename_postfix}.jsonl', 'w', encoding='utf-8') as f:
        for t in triples:
            f.write(f"[{t['qid']},{t["pid+"]},{t["pid-"]}]\n")

    with open(f'{out_path}/meta_{filename_postfix}.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(meta))

    with open(f'{out_path}/qrel_{filename_postfix}', 'w', encoding='utf-8') as f:
        for qrel_entry in qrel:
            f.write(f"{qrel_entry["qid"]} 0 {qrel_entry["pid"]} {qrel_entry["relevance"]}\n")

    
    query_avglen /= samples_ctr
    doc_avglen /= doc_ctr

    print(f"Dataset [{filename_postfix}] with")
    print(f"Query max words ~ {query_maxlen},      Query avg words ~ {query_avglen}")
    print(f"Document max words ~ {doc_maxlen},      Document avg words ~ {doc_avglen}")


def main(data_path, out_path, train_p, min_answers, max_answers):
    data = json.load(open(f'{data_path}/cleaned_with_links.json', encoding='utf-8'))
    makedirs(out_path, exist_ok=True)

    # Split train and test parts
    train, test = train_test_split(data, train_size=train_p)
    create_dataset(train, out_path, "train", min_answers, max_answers)
    create_dataset(test, out_path, "test", min_answers, max_answers)



if __name__ == "__main__":
    data_path = "../ARQMath/data_preprocessing/"
    out_path = "../ARQMathAgg/dataset_2_4/"
    train_p = 0.7  # --> valid_p = 1 - train_p, no test set, because ARQMath provides test set

    # Configuration for concatenation
    min_answers = 2
    max_answers = 3  # Randomize the number of answers concatenated

    main(data_path, out_path, train_p, min_answers, max_answers)
