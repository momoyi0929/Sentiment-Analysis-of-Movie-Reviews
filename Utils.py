import csv
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')
import math
class Utils:
    """
    This class contains the relevant functions required by the Bayesian model
    e.g. load_data, compute_tfidf ...
    """

    def load_data(self, filename):
        """
        load_data:
            load 'test','dev' and 'train' data even though dev data
            train data structures differ from test data(no label)
        ----------
        :param filename: the path to the loading file
        :return:  Dict[str, List[Union[str, int]]] = {}
           e.g.  [id, [review_sentence, segment_label]]
        """
        # make reviews for saving data
        reviews = {}
        # open TSV files
        with open(filename, mode='r', encoding='utf-8') as file:
            tsv_reader = csv.reader(file, delimiter='\t')
            next(tsv_reader)
            # Iterate through each line in the file
            for row in tsv_reader:
                review_ids = row[0]
                review = row[1]
                # We assign -1(int) as the segment of test data for unified format
                # But we won't use it
                if filename not in ['moviereviews/dev.tsv', 'moviereviews/train.tsv']:
                    # process 'test data'
                    if review_ids not in reviews.keys():
                        reviews[review_ids] = [review,-1]
                else:
                    Sentiment = row[2]
                    if review_ids not in reviews.keys():
                        if row[-1] in ['0', '1', '2', '3', '4']:
                            reviews[review_ids] = [review, int(Sentiment)]
        return reviews

    def decide_models(self, filename, class_value):
        """
        decide_models:
            preprocess review sentences (including 'test data')
            (e.g. lowercasing,word_tokenize, stoplist etc.)
            provide datas structures in different modes by 'class_value'
            (5-value, 3-value)
        ----------
        :param filename, class_value: class_value(int) decides class model
        Return: Dict[str, List[Union[str, int]]] = self.load_data(filename
        """
        reviews = self.load_data(filename)
        # decide the model
        if class_value == 3 and filename != 'moviereviews/test.tsv':
            for id, [_,label] in reviews.items():
                if label < 2:
                    reviews[id][1] = 0
                elif label > 2:
                    reviews[id][1] = 2
                else:
                    reviews[id][1] = 1
        # preprocess the data
        for id, [review, _] in reviews.items():
            reviews[id][0] = self.preprocessing(review)
        return reviews

    def preprocessing(self, raw_text):
        """
        :param raw_text: e.g. 'An empty , purposeless exercise .'
        :return: ['empty', 'purpose', 'exercise']
        """
        raw_text = raw_text.lower()
        # segment
        tokens = nltk.word_tokenize(raw_text)

        # Remove special characters and numbers
        characters = [',', '.', '`', '``', '\'',
                      '\'\'', '-', '--', ':',
                      ';', '-lrb-', '-rrb-']
        tokens = [word for word in tokens if word not in characters ]
        tokens = [word for word in tokens if not any(char in string.punctuation or char.isdigit() for char in word)]

        # Word form normalization
        lematizer = WordNetLemmatizer()
        words = [lematizer.lemmatize(raw_word) for raw_word in tokens]

        # Remove stop words
        filtered_text = [word for word in words if word not in stopwords.words('english')]

        return filtered_text

    def compute_tfidf(self, reviews, class_value):
        """
        compute_tfidf:
            use compute_tf and compute_idf to calculate the tfidf value for each word
            consider the different class and provide the tfidf value for each class
        ----------
         TF = The number of times term T appears in document D/the total number of terms in document D
         IDF = idf = log (total number of documents/number of documents containing T terms + 1)
        ----------
        :param reviews: e.g. ['entertaining', 'inferior']
        :param class_value: 3/5 (int)
        :return: tfidf_list: return the tfidf of words for each class
         e.g. In 3-value: {'entertaining': [0, 0.7985462791987848, 0], 'yield': [0, 0.inferior, 0]}
        :return: tfidf_totals: return sum of tfidf of all tokens in each class
         e.g. In 3-value: {0:98730:25432423, 1:42342.67543535, 2:78979:32131444}
        """
        documents = [review for _, (review, _) in reviews.items()]
        labels = [label for _, (_, label) in reviews.items()]

        # calculate TF
        tf_per_doc = [self.compute_tf(doc) for doc in documents]
        # calculate IDF
        idf = self.compute_idf(documents)
        # calculate TF-IDF
        tfidf = {}
        tfidf_totals = {}
        for label in set(labels):
            tfidf_totals[label] = 0
        for doc_idx, tf in enumerate(tf_per_doc):
            label = labels[doc_idx]
            for word, tf_val in tf.items():
                if word not in tfidf:
                    tfidf[word] = {}
                if label not in tfidf[word]:
                    tfidf[word][label] = 0
                tfidf[word][label] += tf_val * idf[word]
                tfidf_totals[label] += tfidf[word][label]
        # calculate tfidf value for each class
        for i in range (class_value):
            for tokens,label_value in tfidf.items():
                if i not in label_value:
                    label_value[i] = 0
        # Convert dictionary format to list format
        tfidf_list = {}
        for word, label_dict in tfidf.items():
            tfidf_list[word] = [label_dict.get(i, 0) for i in range(len(label_dict))]

        return tfidf_list, tfidf_totals

    def compute_tf(self, review):
        """
        :param review: [token1,token2...]
        :return: tf value: {token1: tf, token2: tf}
        """
        tf_dict = {}
        doc_length = len(review)
        for word in review:
            if word not in tf_dict:
                tf_dict[word] = 0
            tf_dict[word] += 1
        for word in tf_dict:
            tf_dict[word] = tf_dict[word] / doc_length
        return tf_dict

    def compute_idf(self, reviews):
        """
        :param reviews: review: [token1,token2...]
        :return: tf value: {token1: tf, token2: tf}
        """
        N = len(reviews)
        idf_dict = {}
        for review in reviews:
            for word in set(review):
                if word not in idf_dict:
                    idf_dict[word] = 0
                idf_dict[word] += 1
        for word in idf_dict:
            idf_dict[word] = math.log(N / float(idf_dict[word]))

        return idf_dict
