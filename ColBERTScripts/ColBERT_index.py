import os

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

    dataroot = '../ARQMathAgg/dataset/'
    dataset_fraction = 'test'
    collection = os.path.join(dataroot, f'collection_{dataset_fraction}.tsv')


    nbits = 2   # encode each dimension with 2 bits
    doc_maxlen = 300   # truncate passages at 300 tokens

    checkpoint = '/home/past12am/Projects/MathIR/ColBERTCheckpoints/colbertv2.0'
    index_name = f'arqmath.{dataset_fraction}.{nbits}bits'

    with Run().context(RunConfig(nranks=1, experiment="ColBERTTestARQMath")):

        config = ColBERTConfig(
            nbits=nbits,
            doc_maxlen=doc_maxlen,
            kmeans_niters=10
        )


        print("Indexing")
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)
