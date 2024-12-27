# MathIR
Note: for all imports to work always run the python scripts from within the directory they are located at

## Build Dataset

1. Download ARQMath from https://drive.google.com/drive/folders/1YekTVvfmYKZ8I5uiUMbs21G2mKwF9IAm into ```ARQMath/raw/```

2. Clone ARQMath Code Repo: https://github.com/ARQMath/ARQMathCode into DatasetProcesssing/ARQMathCode
```git clone https://github.com/ARQMath/ARQMathCode```

4. Run ``get_clean_json.py``

5. Set configuration in ``create_aggregate_dataset.py`` and run the script


## ColBERT

### Training

### Indexing
To index the collection_test.tsv part of the dataset set the paths in and run the file ```ColBERT_index.py``` located in ```ColBERTScripts/```. This will take quite some memory and time.

### Evaluation
