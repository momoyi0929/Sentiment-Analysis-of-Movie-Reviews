# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse
import csv

import numpy as np

from Naive_Bayes_model import Naive_Bayes_model
from Utils import Utils

import seaborn as sns
import matplotlib.pyplot as plt


"""
IMPORTANT, modify this part with your details
"""
USER_ID = "acf21yl" #your unique student ID, i.e. the IDs starting with "acp", "mm" etc that you use to login into MUSE

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args


def main():
    
    inputs=parse_args()
    
    #input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    
    #number of classes
    number_classes = inputs.classes
    
    #accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features

    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
     
    
    #whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix
    
    """
    ADD YOUR CODE HERE
    Create functions and classes, using the best practices of Software Engineering
    """

    # Initialize the bayes model and relevant functions
    # Preprocess the train data and dev data
    utils = Utils()
    a = utils.decide_models(training, number_classes)
    b = utils.decide_models(dev, number_classes)
    nb_model = Naive_Bayes_model()
    nb_model.training_data(a, number_classes, features)
    nb_model.prediction(b, number_classes, features)
    prediction = nb_model.label_predictions
    N = number_classes

    # define the confusion matrix
    matrix = [[0 for _ in range(N)] for _ in range(N)]
    # count in the matrix
    for id, label in prediction.items():
        matrix[label][b[id][1]] += 1
    # decide the printing width
    width = max(len(str(num)) for row in matrix for num in row) + 1
    # Define the confusion matrix
    # Calculate TP, FP, FN for each class
    TP = np.diag(matrix)
    FP = np.sum(matrix, axis=0) - TP
    FN = np.sum(matrix, axis=1) - TP
    # Calculate precision, recall, and F1 score for each class
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    # Calculate macro F1 score
    macro_f1 = np.mean(f1)
        # uncomment this if need more info
        # print('precision: ',precision,'\n', 'recall',recall,'\n','f1', f1,'\n','micro_f1', micro_f1,'\n')

    if output_files:
        # 3-value best model: features; 5-value best model: all_words
        if number_classes == 3:
            if features == 'all_words':
                nb_model.prediction(b, number_classes, 'features')
                prediction = nb_model.label_predictions

            c = utils.decide_models(test, number_classes)
            nb_model.prediction(c, number_classes, 'features')
            test_prediction = nb_model.label_predictions

        else:
            if features == 'features':
                nb_model.prediction(b, number_classes, 'all_words')
                prediction = nb_model.label_predictions

            c = utils.decide_models(test, number_classes)
            nb_model.prediction(c, number_classes, 'all_words')
            test_prediction = nb_model.label_predictions

        file_name = f"dev_predictions_{number_classes}classes_{USER_ID}.tsv"
        output_file = [["SentenceID", "Sentiment"]] + [[id, label] for id, label in prediction.items()]
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerows(output_file)

        file_name_test = f"test_predictions_{number_classes}classes_{USER_ID}.tsv"
        output_file_test = [["SentenceID", "Sentiment"]] + [[id, label] for id, label in test_prediction.items()]
        with open(file_name_test, 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerows(output_file_test)

    #You need to change this in order to return your macro-F1 score for the dev set
    f1_score = macro_f1

    draw = 0
    if confusion_matrix:
        draw = 1
        # Print aligned matrix
        for row in matrix:
            print(" ".join(f"{num:>{width}}" for num in row))

    """
    IMPORTANT: your code should return the lines below. 
    However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
    """
    #print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))
    # use seaborn to draw matrix 
    if draw == 1:
        sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu")
        plt.ylabel("True label")
        plt.xlabel("Predict label")
        plt.title(f"Confusion Matrix with {features} and {number_classes}classes \nF1 score: {f1_score}")
        plt.show()

if __name__ == "__main__":
    main()



