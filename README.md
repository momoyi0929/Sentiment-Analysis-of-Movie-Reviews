## Libraries
- `nltk`: nltk library provides tools for preprocessing (e.g. stopwords, normalization)
    - `WordNetLemmatizer`: From `nltk.stem`, use for word form normalization
    - `stopwords`: From `nltk.corpus`, use for removing stop words
- `numpy` `math` `string` `csv` `argparse`: Standard libraries are imported for loading data(`numpy` `math`), math calculation(`csv` `argparse) and data structure process(`string`)
- `seaborn` `matplotlib`: use for drawing confusion matrix

## Features selection
- `TF-IDF`: The project calculates each word TF-IDF value as the word feature and calculate the probability similar with bayes formula 
- `Combination`: Through testing on the dev data set, we found that the probability obtained by multiplying the probability of tf idf and the probability of Naive Bayes is the most accurate. Thus, we take it as our the core features extraction algorithm.

## Command
    python3 NB_sentiment_analyser.py moviereviews/train.tsv moviereviews/dev.tsv moviereviews/test.tsv -classes ... -features ... -confusion_matrix -output_files
        -classes: 3, 5
        -features: features, all_words
        -confusion_matrix, `-output_files`: optional （if no input： default `False`)

## Download
- `pip install seaborn`: download for plotting confusion matrix
- `pip install matplotlib`: download for plotting
- `pip install nltk`: download the preprocessing library
- `nltk.download('wordnet')`: download the dataset for WordNetLemmatizer
- `nltk.download('stopwords')`: download the dataset for stopwords

## Reference
- “Improved Bayes Method Based on TF-IDF Feature and Grade Factor Feature for Chinese Information Classification | IEEE Conference Publication | IEEE Xplore,” ieeexplore.ieee.org. https://ieeexplore.ieee.org/abstract/document/8367204 (accessed Dec. 15, 2023).
‌