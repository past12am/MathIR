import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ColBERT'))

from ColBERT.colbert.infra import Run, RunConfig, ColBERTConfig
from ColBERT.colbert import Searcher



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
