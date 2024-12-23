import json
import random
from random import shuffle
from collections import defaultdict
from tqdm import tqdm
from os import makedirs
import pandas as pd

# Dataset preparation for ColBERT and ALBERT compatibility
# Focus on aggregating questions and answers into documents

# Paths and configuration
data_path = '../data_processing'
out_path = '../task1/training_files'
train_p = 0.9  # Training fraction (valid_p = 1 - train_p)

data = json.load(open(f'{data_path}/cleaned_with_links.json', encoding='utf-8'))
makedirs(out_path, exist_ok=True)

# 1. Remove questions without answers and group questions by tag
questions_with_answers = defaultdict(list)
for q in data:
    if 'answers' not in q:
        continue  # Skip questions without answers
    for tag in q['tags']:
        questions_with_answers[tag].append(q)

# 2. Check the number of questions for each tag
print('Questions with answers, sizes by tag:')
for tag in questions_with_answers:
    print(tag, len(questions_with_answers[tag]))

# 3. Aggregate questions and answers into document-like structures
aggregated_documents = []
for q in tqdm(data):
    if 'answers' in q:
        doc_id = q['post_id']
        content = q['title'] + ' ' + q['question'] + '\n' + '\n'.join(q['answers'])
        aggregated_documents.append({
            'doc_id': doc_id,
            'content': content
        })

# 4. Create correct and incorrect question-answer pairs for training
correct_pairs = []
wrong_pairs = []
for q in tqdm(data):
    if 'answers' in q:
        correct_answer = random.choice(q['answers'])
        correct_pairs.append((q['title'] + ' ' + q['question'], correct_answer, '1'))  # Label 1 for correct pairs
    tag_a = random.choice(q['tags'])
    try:
        d_b = random.choice(questions_with_answers[tag_a])  # Select a question sharing the tag
    except:
        tag_a = random.choice(list(questions_with_answers.keys()))
        d_b = random.choice(questions_with_answers[tag_a])
    while q['post_id'] == d_b['post_id']:
        if len(questions_with_answers[tag_a]) == 1 and len(q['tags']) > 1:
            tag_a = random.choice(q['tags'])
            if len(questions_with_answers[tag_a]) == 1:
                tag_a = random.choice(list(questions_with_answers.keys()))
        elif len(questions_with_answers[tag_a]) == 1:
            tag_a = random.choice(list(questions_with_answers.keys()))
        d_b = random.choice(questions_with_answers[tag_a])
    wrong_answer = random.choice(d_b['answers'])
    wrong_pairs.append((q['title'] + ' ' + q['question'], wrong_answer, '0'))  # Label 0 for incorrect pairs

# 5. Shuffle and split data into train and validation sets
all_pairs = [*correct_pairs, *wrong_pairs]
shuffle(all_pairs)

no_all = len(all_pairs)
no_train = int(no_all * train_p)

# Split data into train and validation
train_pairs = all_pairs[:no_train]
val_pairs = all_pairs[no_train:]

# Save the splits
def save_to_csv(split, data_pairs, output_path):
    df = pd.DataFrame(data_pairs, columns=['question', 'answer', 'label'])
    df.to_csv(f'{output_path}/arqmath_task1_{split}.csv', index_label='idx')

save_to_csv('train', train_pairs, out_path)
save_to_csv('dev', val_pairs, out_path)

# Save aggregated documents for ColBERT and ALBERT
with open(f'{out_path}/arqmath_documents.jsonl', 'w', encoding='utf-8') as f:
    for doc in aggregated_documents:
        f.write(json.dumps(doc) + '\n')

print('Dataset preparation complete.')
