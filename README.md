# MathIR
Note: for all imports to work always run the python scripts from within the directory they are located at

## Download Dataset and ColBERT checkpoints

To download the dataset, checkpoints and indexes got to TODO where you find the directories
```
    ARQMathAgg/
    ColBERTCheckpoints/
```
Download these and put their contents in the appropriate folders in the repository (or anywhere else, but then you'll need to adapt the paths in the scripts)



## Build Dataset
Skip if you downloaded the dataset

1. Download ARQMath from https://drive.google.com/drive/folders/1YekTVvfmYKZ8I5uiUMbs21G2mKwF9IAm into ```ARQMath/raw/```

2. Clone ARQMath Code Repo: https://github.com/ARQMath/ARQMathCode into DatasetProcesssing/ARQMathCode
```git clone https://github.com/ARQMath/ARQMathCode```

4. Run ``get_clean_json.py``

5. Set configuration in ``create_aggregate_dataset.py`` and run the script


## ColBERT
Skip Training and Indexing if you downloaded the checkpoints and indexes

### Training

### Indexing
To index the collection_test.tsv part of the dataset set the paths in and run the file ```ColBERT_index.py``` located in ```ColBERTScripts/```. This will take quite some memory and time.

### Evaluation
Adapt settings in and run ```python MathIR_Eval_Colbert.py``` to generate a ```Evaluation/ColBERT/run``` (optional, if exists) and ```Evaluation/ColBERT/res.json``` files, where the res.json file holds the results for all evaluated metrics.
