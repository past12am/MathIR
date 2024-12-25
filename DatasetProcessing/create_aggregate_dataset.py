# based on https://github.com/AnReu/ALBERT-for-Math-AR/blob/main/preprocessing_scripts/create_training_data_task1.py

import json
import random
from random import shuffle
from collections import defaultdict
from tqdm import tqdm
from os import makedirs
import pandas as pd

# Create training set: <Query, Positive Document, Negative Document> triples

data_path = '../ARQMath/data_preprocessing/'
out_path = '../ARQMathAgg/dataset'
train_p = 0.7  # --> valid_p = 1 - train_p

# Configuration for concatenation
min_answers = 2
max_answers = 5  # Randomize the number of answers concatenated

data = json.load(open(f'{data_path}/cleaned_with_links.json', encoding='utf-8'))
makedirs(out_path, exist_ok=True)

# 1. Remove questions without answers and group by tags
questions_with_answers = defaultdict(list)
for q in data:
    if 'answers' not in q:
        continue
    for tag in q['tags']:
        questions_with_answers[tag].append(q)

# 2. Prepare <Query, Positive Document, Negative Document> triples
triples = []
jsonl_triples = []
queries = []
collection = []
query_id_map = {}
doc_id_map = {}
query_counter = 0
doc_counter = 0

for q in tqdm(data):
    if 'answers' not in q:
        continue

    # Positive document: Randomize the number of concatenated answers
    num_answers = random.randint(min_answers, max_answers)
    positive_answers = random.sample(q['answers'], min(len(q['answers']), num_answers))
    positive_doc = "\n\n".join([f"Section {i+1}:\n{ans}" for i, ans in enumerate(positive_answers)])

    # Track which answer is the correct one
    correct_answer = random.choice(positive_answers)
    correct_idx = positive_answers.index(correct_answer)

    # Assign IDs for query and documents
    if q['post_id'] not in query_id_map:
        query_id_map[q['post_id']] = query_counter
        queries.append({
            'qid': query_counter,
            'query': q['title'] + ' ' + q['question']
        })
        query_counter += 1

    positive_doc_id = doc_counter
    doc_id_map[positive_doc_id] = positive_doc
    collection.append({
        'pid': positive_doc_id,
        'document': positive_doc
    })
    doc_counter += 1

    # Negative document: Select answers from other questions with shared tags
    tag_a = random.choice(q['tags'])
    try:
        d_b = random.choice(questions_with_answers[tag_a])
    except:
        # If no questions available, pick a random tag and question
        tag_a = random.choice(list(questions_with_answers.keys()))
        d_b = random.choice(questions_with_answers[tag_a])

    while q['post_id'] == d_b['post_id']:
        tag_a = random.choice(q['tags'])
        if len(questions_with_answers[tag_a]) == 1:
            tag_a = random.choice(list(questions_with_answers.keys()))
        d_b = random.choice(questions_with_answers[tag_a])

    num_negative_answers = random.randint(min_answers, max_answers)
    negative_answers = random.sample(d_b['answers'], min(len(d_b['answers']), num_negative_answers))
    negative_doc = "\n\n".join([f"Section {i+1}:\n{ans}" for i, ans in enumerate(negative_answers)])

    negative_doc_id = doc_counter
    doc_id_map[negative_doc_id] = negative_doc
    collection.append({
        'pid': negative_doc_id,
        'document': negative_doc
    })
    doc_counter += 1

    # Ensure no overlap between positive and negative documents
    if any(ans in positive_answers for ans in negative_answers):
        continue

    triples.append({
        'query': q['title'] + ' ' + q['question'],
        'positive_doc': positive_doc,
        'negative_doc': negative_doc,
        'correct_idx': correct_idx  # Index of the correct answer in the positive document
    })

    jsonl_triples.append({
        'qid': query_id_map[q['post_id']],
        'pid+': positive_doc_id,
        'pid-': negative_doc_id
    })

# 3. Shuffle and split the data
shuffle(triples)
shuffle(jsonl_triples)
no_all = len(triples)
no_train = int(no_all * train_p)
no_val = no_all - no_train

train_triples = triples[:no_train]
val_triples = triples[no_train:]
train_jsonl = jsonl_triples[:no_train]
val_jsonl = jsonl_triples[no_train:]

def save_split(split, triples, jsonl_data):
    df = pd.DataFrame(triples)
    df.to_csv(f'{out_path}/arqmath_task1_{split}.csv', index_label='idx')
    with open(f'{out_path}/arqmath_task1_{split}.jsonl', 'w', encoding='utf-8') as f:
        for item in jsonl_data:
            f.write(json.dumps(item) + '\n')

def save_queries_and_collection():
    with open(f'{out_path}/queries.tsv', 'w', encoding='utf-8') as f:
        for query in queries:
            f.write(f"{query['qid']}\t{query['query']}\n")

    with open(f'{out_path}/collection.tsv', 'w', encoding='utf-8') as f:
        for doc in collection:
            f.write(f"{doc['pid']}\t{doc['document']}\n")

save_split('train', train_triples, train_jsonl)
save_split('dev', val_triples, val_jsonl)
save_queries_and_collection()

print('Done creating <Query, Positive Document, Negative Document> triples for training and JSONL files.')