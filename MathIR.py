from ColBERTScripts.ColBERTDocRetrieval import ColBERTDocRetrieval


def main():
    colbert_checkpoint = '/home/past12am/Projects/MathIR/ColBERTCheckpoints/colbertv2.0'

    colbert_data_collection = '/home/past12am/Projects/MathIR/ColBERTScripts/downloads/lotte/lifestyle/test/collection.tsv'
    colbert_index_root = '/home/past12am/Projects/MathIR/ColBERTScripts/experiments/ColBERTTest/indexes/'
    colbert_index = 'lifestyle.test.2bits'

    colbert = ColBERTDocRetrieval(colbert_checkpoint, colbert_data_collection, None, colbert_index_root)
    matches = colbert.get_documents_ColBERT("What is baking soda?")

    for match in matches:
        print(match)





if __name__=='__main__':
    main()
