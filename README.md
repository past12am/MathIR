# MathIR

## Setup Dataset

1. Download ARQMath from https://drive.google.com/drive/folders/1YekTVvfmYKZ8I5uiUMbs21G2mKwF9IAm into ARQMath/raw/

2. Clone ARQMath Code Repo: https://github.com/ARQMath/ARQMathCode into DatasetProcesssing/ARQMathCode
```git clone https://github.com/ARQMath/ARQMathCode```

3. Change line 31 to 36 in ``ARQMathCode/post_reader_record.py`` to
    ```
    post_file_path = root_file_path + "/Posts.V"+version+".xml"
    badges_file_path = root_file_path + "/Badges.V"+version+".xml"
    comments_file_path = root_file_path + "/Comments.V"+version+".xml"
    votes_file_path = root_file_path + "/Votes.V"+version+".xml"
    users_file_path = root_file_path + "/Users.V"+version+".xml"
    post_links_file_path = root_file_path + "/PostLinks.V"+version+".xml"
    ```

4. Run ``get_clean_json.py``

5. Set configuration in ``create_aggregate_dataset.py`` and run the script


## ColBERT

### Training

### Indexing
To index the full collection.tsv set the paths in and run the file (this might take )

### Evaluation
