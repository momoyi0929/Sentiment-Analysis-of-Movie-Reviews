from Utils import Utils
import math
import numpy as np

class Naive_Bayes_model:

    def __init__(self):
        self.u= Utils()
        self.prior_probability = []

        self.tokens_frequency = [] # tokens frequency in each class
        self.sentiment_tokens_num = [] # tokens num in each class
        self.distinct_features = 0  # vocabulary number

        self.tfidf = {} #e.g. {'bad': [0.0,0.0], 'character': {0: 0.0, 1: 0.0}, 'time': {0: 0.17328679513998632}}
        self.tfidf_totals = {} #e.g. {0: 0.34657, 1: 0.74653, 2:0.42144}

        self.label_predictions = {} # {id: label}

    def training_data(self, reviews, class_value, features='all_words'):
        """
        training_data:
             is used as bayes training.
             save prior probability and likelihood in the Unit class based on 'train data'.
             has 3-value, 5-value, 'features' and 'all words'(default) models
        """

        # Initialize for avoiding calculation bugs in 'non-feature' modes
        for i in range(class_value):
            self.tfidf[i] = 1
            self.tfidf_totals[i] = 1
        # calculate the tfidf value in advance if it's 'features model'
        if features == 'features':
            self.tfidf, self.tfidf_totals = self.u.compute_tfidf(reviews, class_value)

        # prior probability (under): total num of reviews
        total_num = len(reviews)
        # prior probability (over): review frequency in each class
        review_fre = [0 for _ in range(class_value)]
        # likelihood (under): count words from each class
        token_num = [[] for _ in range(class_value)]
        # likelihood (over): record frequency of word in each class
        token_fre = [{} for _ in range(class_value)]
        # smoothing : vocabulary number
        num_distinct_features = []

        for _, [review, label] in reviews.items():
            review_fre[label] += 1
            for token in review:
                if token in token_num[label]:
                    token_fre[label][token] += 1
                else:
                    token_fre[label][token] = 1
                token_num[label].append(token)
            for token in review:
                if token not in num_distinct_features:
                    num_distinct_features.append(token)

        self.distinct_features = len(num_distinct_features)
        self.prior_probability = [review_fre[i] / total_num
                                  for i in range(class_value)]
        self.sentiment_tokens_num = token_num
        self.tokens_frequency = token_fre

    def prediction(self, test, class_value, features='all_words'):
        """
        prediction:
            loop through each sentence in the test data
            attach the label and save it as key-value in dictionary
        ----------
        :param test: test/dev data
        :param class_value: 3/5 value
        :param features: features/all_words
        :return: {id: label}, the label e.g. 0,1,2,3,4
        """

        self.label_predictions = {} # Initialization avoids saving bug
        for id, [review, _] in test.items():
            predict_label = self.attach_label(review, class_value, features)
            self.label_predictions[id] = self.enhance_feature(predict_label, class_value)

    def attach_label(self, review, class_value, features='all_words'):
        """
        attach_label:
            make use of the value the training_data saved in the Unit class for calculation
            provide the probability of each class for comparing
        -----------
        :return: In 3-value: e.g. [1.324424, 2.4312332, 1.3333233]
                 the probability in class 0, 1, 2
        """

        # initialize tfidf and bayes probability list for comparing
        labels_probability_tfidf = []
        labels_probability = []
        for i in range(class_value):
            # prepare prior probability in advance
            posterior = self.prior_probability[i]
            # initialize tfidf value
            review_tfidf = 0
            # calculate the bayes posterior for the review sentence
            for token in review:
                # check if the training data has this word
                if token in self.tokens_frequency[i]:
                    posterior_over = self.tokens_frequency[i][token]
                else:
                    posterior_over = 0
                # bayes formula
                posterior *= (posterior_over + 1) / \
                             (len(self.sentiment_tokens_num[i]) + self.distinct_features)
                # calculate probability by tfidf in 'features' model
                if features == 'features':
                    if token in self.tfidf:
                        review_tfidf += self.tfidf[token][i]
                    else:
                        # compared with the bayes equation, add 1 for smoothing in tfidf model
                        review_tfidf += 1
            # save the probability for each class
            labels_probability_tfidf.append(review_tfidf / self.tfidf_totals[i])
            labels_probability.append(posterior)

        output = labels_probability
        # combine bayes result with tfidf value
        if features == 'features':
            combination = [i * j for i, j in zip(labels_probability, labels_probability_tfidf)]
            output = combination

        return output

    def enhance_feature(self, result, class_value):
        """
        According to the experience of experiment on different weights of output,
        1.6 seems to be the best one leading to the high macro_f1 grades
        ----------
        :return: label: 0,1,2,3,4
        """
        if class_value == 3:
            result[1] *= 1.6
        if class_value == 5:
            result[0] *= 1.6
            result[2] *= 1.6
        return result.index(max(result))
