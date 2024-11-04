#!/usr/bin/env python

'''TODO: add high-level description of this Python script'''

import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_file", default='data_project/train.tsv', type=str,
                        help="Train file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev_file", default='data_project/dev.tsv', type=str,
                        help="Dev file to evaluate on (default dev.txt)")
    args = parser.parse_args()
    return args


def read_corpus(df):
    """
    This function takes a DataFrame as input, where:
    - The first column is expected to contain the text (documents).
    - The second column contains the labels.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the corpus with text in the first column and labels in the second column.
                          
    Returns:
    tuple: A tuple (documents, labels) where:
           - documents is a list of tokenized text data (list of lists of strings)
           - labels is a list of labels (strings)
    """
    
    # Assign the first column to documents and split text into tokens
    documents = df.iloc[:, 0].fillna("").astype(str).apply(lambda x: x.split()).tolist()
    
    # Assign the second column to labels
    labels = df.iloc[:, 1].tolist()
    
    return documents, labels


def identity(inp):
    '''Dummy function that just returns the input'''
    return inp


if __name__ == "__main__":
    args = create_arg_parser()

    train = pd.read_csv(args.train_file, sep='\t')
    dev = pd.read_csv(args.dev_file, sep='\t')

    # TODO: comment
    X_train, Y_train = read_corpus(train)
    X_test, Y_test = read_corpus(dev)

    # Convert the texts to vectors
    # We use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
 
    vec = CountVectorizer(preprocessor=identity, tokenizer=identity)

    # Combine the vectorizer with a Naive Bayes classifier
    # Of course you have to experiment with different classifiers
    # You can all find them through the sklearn library
    classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])

    # TODO: comment this
    classifier.fit(X_train, Y_train)

    # TODO: comment this
    Y_pred = classifier.predict(X_test)
    

    # TODO: comment this
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, pos_label='OFF')
    precision = precision_score(Y_test, Y_pred, pos_label='OFF')
    recall = recall_score(Y_test, Y_pred, pos_label='OFF')

    print(f"Final accuracy: {acc}")
    print(f"Final precision: {precision}")
    print(f"Final recall: {recall}")
    print(f"Final F1-Score: {f1}")
