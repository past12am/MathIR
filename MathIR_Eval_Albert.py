from pylatexenc.latex2text import LatexNodes2Text
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import numpy as np
from os import listdir
from os.path import join
import re
import ir_measures
from ir_measures import nDCG, P, Judged, MRR, MAP, R
import itertools
from tqdm import tqdm
import os

import pandas as pd
import json

from ALBERTScripts.ALBERTDocRetrieval import ALBERTDocRetrieval

def main():
    parse_all_documents('./ARQMathAgg/dataset_v2/',
                      'collection_test.tsv', 'queries_test.tsv', 'aggregates/collection_agg_test_extended.json', "qrel_test")


def parse_all_documents(dataset_base_path, collection_name, queries_name, meta_name, qrel_name):
    # Configuration

    #   Dataset
    collection_path = join(dataset_base_path, collection_name)
    queries_path = join(dataset_base_path, queries_name)
    meta_path = join(dataset_base_path, meta_name)
    qrelfile = join(dataset_base_path, qrel_name)


    #   ALBERT
    k = 25
    albert_tokenizer = "albert/albert-base-v2"
    albert_classifier = "AnReu/albert-for-math-ar-base-ft"
    


    #   Output
    gen_runfile = False
    break_at = 1000

    eval_res_out_path = f'./Evaluation/ALBERT/{albert_classifier.replace("/", "_")}/'
    runfile = f'{eval_res_out_path}/run'

    try:
        with open(meta_path, 'r', encoding='utf-8') as file:
            meta = json.load(file)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    collection = {}
    with open(collection_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.match(r"(\d+)\s+(.*)", line)
            if match:
                index = int(match.group(1))
                text = match.group(2).strip()
                collection[index] = text


    queries = {}
    with open(queries_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.match(r"(\d+)\s+(.*)", line)
            if match:
                index = int(match.group(1))
                text = match.group(2).strip()
                queries[index] = text

    
    if(gen_runfile):
        os.makedirs(eval_res_out_path, exist_ok=True)

        albert = ALBERTDocRetrieval(albert_tokenizer, albert_classifier)

        with open(runfile, 'w', encoding='utf-8') as f:

            ctr = 0
            for obj in tqdm(meta, total=len(meta)):
                qid = obj['qid']
                pids = obj['pids']
                correct_idx = obj['corr_idx']

                paragraph_texts = [collection.get(pid) for pid in pids]

                query = queries.get(qid)

                top_k = albert.get_top_k_paragraphs(paragraph_texts, query, k)
                paragraph_ids_ranked = [(pids[id], score) for id, score in top_k]

                for rank, (pid, score) in enumerate(paragraph_ids_ranked):
                    f.write(f"{qid} Q0 {pid} {rank} {score} STANDARD\n")

                if(ctr >= break_at):
                    break

                ctr+=1


    qrels = ir_measures.read_trec_qrels(qrelfile)
    runs = ir_measures.read_trec_run(runfile)
    cutoffs = [5, 10, 25]

    ndcg_measure = [nDCG(cutoff=cutoff) for cutoff in cutoffs]
    p_measure = [P(cutoff=cutoff) for cutoff in cutoffs]
    jugded_measure = [Judged(cutoff=cutoff) for cutoff in cutoffs]
    mrr_measure = [MRR(cutoff=cutoff) for cutoff in cutoffs]
    map_measure = [MAP(cutoff=cutoff) for cutoff in cutoffs]

    all_measures = list(itertools.chain(ndcg_measure, p_measure, jugded_measure, mrr_measure, map_measure))

    print("Evaluating on:")
    print(all_measures)

    eval_res = ir_measures.calc_aggregate(all_measures, qrels, runs)
    with open(f"{eval_res_out_path}/res.csv", 'w', encoding='utf-8') as f:
        f.write("Metric,Value\n")
        for measure in all_measures:
            f.write(f"{str(measure)},{eval_res[measure]}\n")


def parse_latex_into_array(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    latex_text = ''.join(lines)

    text = LatexNodes2Text(math_mode='verbatim', strict_latex_spaces=True).latex_to_text(latex_text)
    paragraphs = [para.strip() for para in text.split(".\n\n") if para.strip()]
    return paragraphs

if __name__=='__main__':
    main()