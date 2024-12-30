import sys
sys.path.insert(0, '../ColBERT')

from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer


def train():
    # use 4 gpus (e.g. four A100s, but you can use fewer by changing nway,accumsteps,bsize).
    with Run().context(RunConfig(nranks=1)):
        triples = '/home/past12am/Projects/MathIR/ARQMathAgg/dataset_2_3/triples_train.jsonl'
        queries = '/home/past12am/Projects/MathIR/ARQMathAgg/dataset_2_3/queries_train.tsv'
        collection = '/home/past12am/Projects/MathIR/ARQMathAgg/dataset_2_3/collection_train.tsv'

        checkpoint = '/home/past12am/Projects/MathIR/ColBERTCheckpoints/colbertv2.0'

        # https://jina.ai/news/what-is-colbert-and-late-interaction-and-why-they-matter-in-search/
        config = ColBERTConfig(
            bsize=12, 
            lr=3e-06, 
            warmup=10_000,
            doc_maxlen=512,
            query_maxlen=128,
            dim=128,
            nway=2,
            accumsteps=1,
            similarity='cosine',
            use_ib_negatives=True,
            maxsteps=60_000,
            verbose=3
        )
        
        trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config)
        chk_path = trainer.train(checkpoint=checkpoint)

        print(f"Checkpoint saved to {chk_path}")


if __name__ == '__main__':
    train()