from ColBERTScripts.ColBERTDocRetrieval import ColBERTDocRetrieval


def main():
    colbert_checkpoint = './ColBERTCheckpoints/colbertv2.0'

    colbert_data_collection = './ARQMathAgg/dataset/collection_test.tsv'
    colbert_index_root = './ARQMathAgg/indexes/colbertv2.0/'
    colbert_index = 'arqmath.test.2bits'

    colbert = ColBERTDocRetrieval(colbert_checkpoint, colbert_data_collection, colbert_index, colbert_index_root)
    matches = colbert.get_documents_ColBERT("What is an Abelsch group?")

    for match in matches:
        print(match)





if __name__=='__main__':
    main()
