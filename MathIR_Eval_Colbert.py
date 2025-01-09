
from ColBERTScripts.ColBERTDocRetrieval import ColBERTDocRetrieval
from ColBERT.colbert.data import Queries

import json
import itertools

import ir_measures
from ir_measures import nDCG, P, Judged, MRR, MAP, R

import pandas as pd

from tqdm import tqdm


def main():
    # Configuration

    #   Dataset
    collection_path = './ARQMathAgg/dataset_v2/collection_test.tsv'
    queries_path = './ARQMathAgg/dataset_v2/queries_test.tsv'
    triples_path = './ARQMathAgg/dataset_v2/triples_test.jsonl'

    qrelfile = './ARQMathAgg/dataset_v2/qrel_test'


    #   ColBERT
    colbert_version = "colbertmath4" #"colbertv2.0"

    colbert_checkpoint = f'./ColBERTCheckpoints/{colbert_version}'

    colbert_index_root = f'./ARQMathAgg/indexes/{colbert_version}/'
    colbert_index = 'arqmath.test.2bits'


    #   Output
    eval_res_out_path = f'./Evaluation/ColBERT/{colbert_version}/'


    #   Settings
    gen_run_file = True
    k = 25
    cutoffs = [5,10,25]
    break_at = 20000



    # Generate Run File
    runfile = f'{eval_res_out_path}/run'
    if(gen_run_file):
        # Load ColBERT
        colbert = ColBERTDocRetrieval(colbert_checkpoint, collection_path, colbert_index, colbert_index_root)


        # Load queries and triples
        queries = Queries(queries_path)
        triples_df = pd.read_json(triples_path, lines=True)


        # Generate resulst file (https://cs.usm.maine.edu/~behrooz.mansouri/courses/Slides_IR_22/Introduction%20to%20Information%20Retrieval%20--%20Session%206%20-%20Evaluation%20Measures.pdf)
        with open(runfile, 'w', encoding='utf-8') as f:

            ctr = 0
            for ((query_qid, query)) in tqdm(queries, total=(len(queries) if break_at is None else break_at)):

                matches = colbert.get_documents_ColBERT(query, k=k)

                for match in matches:
                    f.write(f"{query_qid} Q0 {match["pid"]} {match["rank"]} {match["score"]} STANDARD\n")

                ctr += 1
                if(break_at is not None and ctr >= break_at):
                    break
                


    # Evaluate run
    qrels = ir_measures.read_trec_qrels(qrelfile)
    runs = ir_measures.read_trec_run(runfile)
    

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
    

if __name__=='__main__':
    main()
