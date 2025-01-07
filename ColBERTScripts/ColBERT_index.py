import os
import datetime

import sys
sys.path.insert(0, '../ColBERT')

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher



# To make CUDA execute work
#   1.) ensure CUDA_HOME is set
#   2.) set GCC compiler to a compatible version: set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc-13")
#           export NVCC_CCBIN='gcc-13'
#   3.) (don't) instead of 2): keep gcc 14 and skip checks
#           export NVCC_PREPEND_FLAGS='--allow-unsupported-compiler'
# 


if __name__=='__main__':
    """
    Note: be careful, for Real
        You should be careful about having global statements, that are not guarded with an if __name__ == '__main__'. If a different start method than fork is used, they will be executed in all subprocesses.
    """

    dataroot = '../ARQMathAgg/dataset_v2/'
    dataset_fraction = 'test'
    collection = os.path.join(dataroot, f'collection_{dataset_fraction}.tsv')

    colbert_checkpoint = "colbertmath4" #"colbertv2.0"

    nbits = 2
    doc_maxlen = 512
    query_maxlen = 128

    checkpoint = f'/home/past12am/Projects/MathIR/ColBERTCheckpoints/{colbert_checkpoint}'
    index_name = f'arqmath.{dataset_fraction}.{nbits}bits'

    with Run().context(RunConfig(nranks=1, experiment=f"exp_{colbert_checkpoint}_{datetime.datetime.now()}")):

        config = ColBERTConfig(
            nbits=nbits,
            doc_maxlen=doc_maxlen,
            query_maxlen=query_maxlen,
            kmeans_niters=10
        )


        print("Indexing")
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)
