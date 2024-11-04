import re
import csv
import os
import argparse
import random
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

np.random.seed(42)
random.seed(42)


def preprocess_text(text):
    '''Cleans and tokenizes the text.'''
    if not isinstance(text, str):
        text = ""  # replace NaNs or non-string entries with an empty string
    text = text.lower()
    return text.split()


def get_word2vec_embeddings(sentences, vector_size=100, window=5, min_count=1):
    '''Trains a Word2Vec model and returns vector embeddings for sentences.'''
    w2v_model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    return w2v_model


def tweet_to_embedding(tweet, w2v_model, vector_size):
    '''Converts a tweet into an averaged Word2Vec embedding.'''
    words = preprocess_text(tweet)
    embeddings = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(vector_size)


def transform_text_data(data, w2v_model, vector_size):
    '''Transforms all tweets in the dataset to their respective embeddings.'''
    return np.array([tweet_to_embedding(tweet, w2v_model, vector_size) for tweet in data['tweet']])


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", action='store_true', help="Makes it so the model also predicts on the test files.")
    parser.add_argument("-nt", "--non_text", action='store_true', help="Choose to use Non-Text features")
    
    subparser = parser.add_subparsers(dest="algorithm", required=True, help="Choose the classifying algorithm to use")
    svr_parser = subparser.add_parser("svc", help="Use Support Vector Classifier")
    svr_parser.add_argument("-k", "--kernel", choices=["linear", "poly", "rbf", "sigmoid"], default="rbf",
                           help="Choose the kernel for the SVC")
    svr_parser.add_argument("-d", "--degree", default=3, type=int,
                               help="ONLY FOR POLY KERNEL, changes the degree of the polynomial kernel function")
    svr_parser.add_argument("-g", "--gamma", choices=["scale", "auto"], default="scale",
                           help="Choose the gamma (kernel coefficient for rbf, poly and sigmoid) for the SVC")
    svr_parser.add_argument("-C", "--C", type=float, default=1, help="Set the regularization parameter")
    
    args = parser.parse_args()
    return args


def choose_model(args):
    if args.algorithm == "svc":
        alg_name = "Support Vector Classifier"
        pred_path = 'predictions/svc.csv'
        model = SVC(kernel=args.kernel, degree=args.degree, gamma=args.gamma, C=args.C, class_weight={"NOT": 0.7470703125, "OFF": 1.5118577075098814})
    return pred_path, model, alg_name


def main():
    
    args = create_arg_parser()
    pred_path, model, algorithm_name = choose_model(args)

    # Read the data
    train = pd.read_csv("data/train.csv")
    dev = pd.read_csv("data/dev.csv")
    test = pd.read_csv("data/test.csv")

    # Preprocess and prepare Word2Vec embeddings
    all_sentences = train['tweet'].apply(preprocess_text).tolist()
    vector_size = 100  # You can adjust this
    w2v_model = get_word2vec_embeddings(all_sentences, vector_size=vector_size)

    # Convert tweets to Word2Vec embeddings
    X_train = transform_text_data(train, w2v_model, vector_size)
    X_dev = transform_text_data(dev, w2v_model, vector_size)
    X_test = transform_text_data(test, w2v_model, vector_size)

    y_train = train["label"]
    y_dev = dev["label"]
    y_test = test["label"]

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.transform(X_dev)
    X_test = scaler.transform(X_test)

    # Train the SVM model
    model.fit(X_train, y_train)

    # Evaluate on dev set
    y_pred = model.predict(X_dev)
    accuracy = accuracy_score(y_dev, y_pred)
    precision = precision_score(y_dev, y_pred, pos_label="OFF", zero_division=0)
    recall = recall_score(y_dev, y_pred, pos_label="OFF")
    f1 = f1_score(y_dev, y_pred, pos_label="OFF")

    print(f'Results for {algorithm_name} on the Development set:')
    print(f'Accuracy: {round(accuracy, 3)}')
    print(f'Precision: {round(precision, 3)}')
    print(f'Recall: {round(recall, 3)}')
    print(f'F1-Score: {round(f1, 3)}')

    # Evaluate on test set if specified
    if args.test:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label="OFF", zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label="OFF")
        f1 = f1_score(y_test, y_pred, pos_label="OFF")
    
        print(f'\nResults for {algorithm_name} on the Test set:')
        print(f'Accuracy: {round(accuracy, 3)}')
        print(f'Precision: {round(precision, 3)}')
        print(f'Recall: {round(recall, 3)}')
        print(f'F1-Score: {round(f1, 3)}')

    # Save predictions
    if args.test:
        data_pred = np.array([y_pred, y_test])
    else:
        data_pred = np.array([y_pred, y_dev])

    dataset = pd.DataFrame({'Predictions': data_pred[0], 'Actual_Values': data_pred[1]})
    dataset.to_csv(pred_path, index=False)


if __name__ == "__main__":
    main()
