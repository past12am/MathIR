import sys
sys.path.insert(0, 'ColBERT')

from ColBERT.colbert.infra import Run, RunConfig, ColBERTConfig
from ColBERT.colbert.data import Queries, Collection
from ColBERT.colbert import Indexer, Searcher


class ColBERTDocRetrieval:
    # based on https://github.com/stanford-futuredata/ColBERT/blob/main/server.py

    def __init__(self, checkpoint, collection, index_name, index_root):
        config = ColBERTConfig(
            nranks=1
        )
        self.searcher = Searcher(checkpoint=checkpoint, collection=collection, index=index_name, index_root=index_root, config=config)

    def get_documents_ColBERT(self, query, k=10):
        pids, ranks, scores = self.searcher.search(query, k=k)

        top_k_docs = list()
        for pid, rank, score in zip(pids, ranks, scores):
            text = self.searcher.collection[pid]
            d = {'text': text, 'pid': pid, 'rank': rank, 'score': score}
            top_k_docs.append(d)

        top_k_docs = list(sorted(top_k_docs, key=lambda p: (-1 * p['score'], p['pid'])))
        return top_k_docs



def main():
    colbert_checkpoint = '/home/past12am/Projects/MathIR/ColBERTCheckpoints/colbertv2.0'

    colbert_data_collection = '/home/past12am/Projects/MathIR/ColBERTScripts/downloads/lotte/lifestyle/test/collection.tsv'
    colbert_index_root = '/home/past12am/Projects/MathIR/ColBERTScripts/experiments/ColBERTTest/indexes/'
    colbert_index = 'lifestyle.test.2bits'

    colbert = ColBERTDocRetrieval(colbert_checkpoint, colbert_data_collection, colbert_index, colbert_index_root)
    matches = colbert.get_documents_ColBERT("What is baking soda?")

    for match in matches:
        print(match)





if __name__=='__main__':
    main()
