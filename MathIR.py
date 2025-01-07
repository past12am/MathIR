from ColBERTScripts.ColBERTDocRetrieval import ColBERTDocRetrieval
from ColBERT.colbert.data import Queries, Collection

from ALBERTScripts.ALBERTDocRetrieval import ALBERTDocRetrieval


import json


def main():
    # Configuration

    #   Dataset
    root_path = './ARQMathAgg/dataset_v2'
    dataset_fraction = 'test'

    collection_path = f'{root_path}/collection_{dataset_fraction}.tsv'

    agg_path = f'{root_path}/aggregates/collection_agg_{dataset_fraction}.json'



    #   ColBERT
    colbert_version = "colbertmath4"

    colbert_checkpoint = f'./ColBERTCheckpoints/{colbert_version}'

    colbert_index_root = f'./ARQMathAgg/indexes/{colbert_version}/'
    colbert_index = 'arqmath.test.2bits'

    k_colbert = 25

    
    #   ALBERT
    albert_tokenizer = "albert/albert-base-v2"
    albert_classifier = "AnReu/albert-for-math-ar-base-ft"

    k_albert = 5


    #   General
    perform_doc_aggregation = False





    # Load ColBERT
    colbert = ColBERTDocRetrieval(colbert_checkpoint, collection_path, colbert_index, colbert_index_root)


    # Load ALBERT
    albert = ALBERTDocRetrieval(albert_tokenizer, albert_classifier)


    # Load collection and document aggregates
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


    
    while (True):

        print("\nInput your Query: ")
        query = input()
        print("\n")

        # ColBERT
        matches_subdoc = colbert.get_documents_ColBERT(query, k=k_colbert)
        


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
        top_k_subdocs = albert.get_top_k_paragraphs(colbert_retrieved_docs_text, query, k_albert)
        


        print("\nResults:")
        for rank, (list_idx, score) in enumerate(top_k_subdocs):
            print(f"[{rank} ({score})]:   {colbert_retrieved_docs_text[list_idx]}\n")

    

if __name__=='__main__':
    main()
