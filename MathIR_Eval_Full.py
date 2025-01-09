
from ColBERTScripts.ColBERTDocRetrieval import ColBERTDocRetrieval
from ColBERT.colbert.data import Queries, Collection

from ALBERTScripts.ALBERTDocRetrieval import ALBERTDocRetrieval

import os
import time

import json
import itertools

import ir_measures
from ir_measures import nDCG, P, Judged, MRR, MAP, R

import pandas as pd

from tqdm import tqdm


def main():
    # Configuration

    #   Dataset
    root_path = './ARQMathAgg/dataset_v2'
    dataset_fraction = 'test'

    collection_path = f'{root_path}/collection_{dataset_fraction}.tsv'
    queries_path = f'{root_path}/queries_{dataset_fraction}.tsv'

    agg_path = f'{root_path}/aggregates/collection_agg_{dataset_fraction}.json'


    qrelfile = './ARQMathAgg/dataset_v2/qrel_test'


    #   ColBERT
    colbert_version = "colbertmath4" #"colbertv2.0"

    colbert_checkpoint = f'./ColBERTCheckpoints/{colbert_version}'

    colbert_index_root = f'./ARQMathAgg/indexes/{colbert_version}/'
    colbert_index = 'arqmath.test.2bits'

    k_colbert = 25

    
    #   ALBERT
    albert_tokenizer = "albert/albert-base-v2"
    albert_classifier = "AnReu/albert-for-math-ar-base-ft"  # "albert/albert-base-v2"

    albert_version = f"{albert_tokenizer}_{albert_classifier}".replace("/", "-")

    k_albert = 25



    #   Output
    eval_res_out_path = f'./Evaluation/Full/{colbert_version}_{albert_version}/'



    #   Evaluation
    gen_run_file = True
    break_at = 10000
    cutoffs = [5,10,25]

    perform_doc_aggregation = False         # false: ALBERT as a re-ranker,     true: ALBERT as selection/ranking of subset (performance issue)



    # Generate Run File
    runfile = f'{eval_res_out_path}/run'
    if(gen_run_file):
        # Load ColBERT
        colbert = ColBERTDocRetrieval(colbert_checkpoint, collection_path, colbert_index, colbert_index_root)

        # Load ALBERT
        albert = ALBERTDocRetrieval(albert_tokenizer, albert_classifier)


        # Load queries, collection and document aggregates
        queries = Queries(queries_path)
        collection = Collection(collection_path)

        agg_dicts = None
        with open(agg_path, 'r', encoding='utf-8') as fa:
            agg_dicts = json.load(fa)

        pid_to_agg_dict = dict()
        for agg in agg_dicts:
            agg["pids"] = tuple(agg["pids"])    # To make it hashable for later use

            for pid in agg["pids"]:

                if(pid not in pid_to_agg_dict.keys()):
                    pid_to_agg_dict[pid] = [agg["agg_id"]]
                else:
                    pid_to_agg_dict[pid].append(agg["agg_id"])


        



        # Generate resulst file (https://cs.usm.maine.edu/~behrooz.mansouri/courses/Slides_IR_22/Introduction%20to%20Information%20Retrieval%20--%20Session%206%20-%20Evaluation%20Measures.pdf)
        os.makedirs(eval_res_out_path, exist_ok=True)
        with open(runfile, 'w', encoding='utf-8') as f:

            ctr = 0

            total_albert_time = 0
            total_samples_eval_albert = 0

            total_colbert_time = 0

            for ((query_qid, query)) in tqdm(queries, total=(len(queries) if break_at is None else break_at)):

                # ColBERT
                t0 = time.time()
                matches_subdoc = colbert.get_documents_ColBERT(query, k=k_colbert)
                t1 = time.time()

                total_colbert_time += t1-t0
                


                # Collect documents from subdocs
                colbert_retrieved_docs_text = list()
                colbert_retrieved_docs_pids = list()
                if(perform_doc_aggregation):
                    
                    colbert_retrieved_docs = dict()
                    colbert_retrieved_docs_dummy = set()

                    for subdoc in matches_subdoc:
                        # "{'text': '$2\\\\sin(2x)\\\\cos(x)=\\\\sin(3x)+\\\\sin(x)$ Use this.', 'pid': 126791, 'rank': 1, 'score': 119.9375}"

                        try:
                            subdoc_agg_ids = pid_to_agg_dict[subdoc["pid"]]

                            for subdoc_agg_id in subdoc_agg_ids:
                                colbert_retrieved_docs[subdoc_agg_id] = (agg_dicts[subdoc_agg_id]["pids"])

                        except KeyError:
                            # due to the dataset split there might be documents retrieved that are not in the aggregation file (randomness), add these to a dummy document
                            colbert_retrieved_docs_dummy.add(subdoc["pid"])
                    
                    colbert_retrieved_docs[-1] = colbert_retrieved_docs_dummy

                
                
                    for pid_list in colbert_retrieved_docs.values():
                        for pid in pid_list:
                            colbert_retrieved_docs_text.append(collection[pid])
                            colbert_retrieved_docs_pids.append(pid)
                    
                else:
                    for subdoc in matches_subdoc:
                        colbert_retrieved_docs_text.append(collection[subdoc["pid"]])
                        colbert_retrieved_docs_pids.append(subdoc["pid"])



                # ALBERT
                t0 = time.time()
                top_k_subdocs = albert.get_top_k_paragraphs(colbert_retrieved_docs_text, query, k_albert)
                t1 = time.time()

                total_albert_time += t1-t0
                total_samples_eval_albert += len(colbert_retrieved_docs_text)
                


                for rank, (list_idx, score) in enumerate(top_k_subdocs):
                    f.write(f"{query_qid} Q0 {colbert_retrieved_docs_pids[list_idx]} {rank} {score} STANDARD\n")

                ctr += 1
                if(break_at is not None and ctr >= break_at):
                    break
                

    num_iter = len(queries) if break_at is None else break_at
    print("Average Retrieval Times for:")
    print(f"ColBERT[k={k_colbert}]: {total_colbert_time/num_iter}")
    print(f"ALBERT[k={total_samples_eval_albert/num_iter} (avg)]: {total_albert_time/num_iter}")


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
    
    with open(f"{eval_res_out_path}/res.json", 'w', encoding='utf-8') as f:
        f.write("Metric,Value\n")
        for measure in all_measures:
            f.write(f"{str(measure)},{eval_res[measure]}\n")
    

if __name__=='__main__':
    main()
