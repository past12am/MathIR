
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
    collection_path = './ARQMathAgg/dataset/collection_test.tsv'
    queries_path = './ARQMathAgg/dataset/queries_test.tsv'
    triples_path = './ARQMathAgg/dataset/triples_test.jsonl'

    qrelfile = './ARQMathAgg/dataset/qrel_test'


    #   ColBERT
    colbert_checkpoint = './ColBERTCheckpoints/colbertv2.0'

    colbert_index_root = './ARQMathAgg/indexes/colbertv2.0/'
    colbert_index = 'arqmath.test.2bits'


    #   Output
    eval_res_out_path = './Evaluation/ColBERT/colbertv2.0/'


    #   Settings
    gen_run_file = False
    k = 50
    cutoffs = [5,10,50]



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

            for ((query_qid, query), (qid, pid_pos, pid_neg)) in tqdm(zip(queries, triples_df.itertuples(index=False)), total=len(queries)):
                assert(query_qid == qid)

                matches = colbert.get_documents_ColBERT(query, k=k)

                for match in matches:
                    f.write(f"{query_qid} Q0 {match["pid"]} {match["rank"]} {match["score"]} STANDARD\n")


    # Evaluate run
    qrels = ir_measures.read_trec_qrels(qrelfile)
    run = ir_measures.read_trec_run(runfile)

    ndcg_measure = [nDCG(cutoff=cutoff) for cutoff in cutoffs]
    p_measure = [P(cutoff=cutoff) for cutoff in cutoffs]
    jugded_measure = [Judged(cutoff=cutoff) for cutoff in cutoffs]
    mrr_measure = [MRR(cutoff=cutoff) for cutoff in cutoffs]
    map_measure = [MAP(cutoff=cutoff) for cutoff in cutoffs]

    all_measures = list(itertools.chain(ndcg_measure, p_measure, jugded_measure, mrr_measure, map_measure))
    
    print("Evaluating on:")
    print(all_measures)

    eval_res = ir_measures.calc_aggregate(all_measures, qrels, run)
    
    with open(f"{eval_res_out_path}/res.json", 'w', encoding='utf-8') as f:
        f.write("Metric,Value")
        for measure in eval_res:
            f.write(f"{str(measure)},{eval_res[measure]}\n")
    

if __name__=='__main__':
    main()
