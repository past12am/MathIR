import sys
sys.path.insert(0, '../ColBERT')

from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer


def train():
    # use 4 gpus (e.g. four A100s, but you can use fewer by changing nway,accumsteps,bsize).
    with Run().context(RunConfig(nranks=1)):
        triples = '/home/past12am/Projects/MathIR/ARQMathAgg/dataset/triples_train.jsonl'
        queries = '/home/past12am/Projects/MathIR/ARQMathAgg/dataset/queries.tsv'
        collection = '/home/past12am/Projects/MathIR/ARQMathAgg/dataset/collection.tsv'

        checkpoint = '/home/past12am/Projects/MathIR/ColBERTCheckpoints/colbertv2.0'

        config = ColBERTConfig(
            bsize=8, 
            lr=1e-05, 
            warmup=20_000,
            doc_maxlen=180,
            dim=128,
            attend_to_mask_tokens=False,
            nway=2,
            accumsteps=1,
            similarity='cosine',
            use_ib_negatives=True,
            maxsteps=500
        )
        
        trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config)
        chk_path = trainer.train(checkpoint=checkpoint)  # or start from scratch, like `bert-base-uncased`

        print(f"Checkpoint saved to {chk_path}")


if __name__ == '__main__':
    train()