#
#
#
#

from os import remove
import re
import argparse

import pandas as pd
import numpy as np
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split

def create_arg_parser():
    '''Creates an argument parser to read the command line arguments.
    This includes subparsers for the different models.
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data", choices=["dev.tsv", "test.tsv", "train.tsv"], default="train.tsv",
                           help="Choose data for processing")
    parser.add_argument("-ht", "--hashtags", action='store_true',
                            help="Choose to extract text or hashtags as features")

    args = parser.parse_args()

    return args


def read_write_data(file):

    path = "data_project/" + file
    column_names = ['tweet', 'label']
    df = pd.read_csv(path, sep='\t', dtype=str, names=column_names, header=None)

    return df

def get_text(text):
    # Regex patterns for removing @ mentions, hashtags, and emojis
    cleaned_text = re.sub(r'@\w+', '', text)  # Remove mentions
    cleaned_text = re.sub(r'#\w+', '', cleaned_text)  # Remove hashtags
    cleaned_text = re.sub(r'[^\w\s,.!?\'"]', '', cleaned_text)  # Remove emojis and special characters
    cleaned_text = re.sub(r'URL', '', cleaned_text)  # Remove emojis and special characters

    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def create_word2vec(df, columns, vector_size=100, window=5, min_count=1):
    '''Takes a list a df and list of columns as input and creates Word2Vec representations for the specified columns'''
    for column in columns:
        tokenized_column = df[column].apply(lambda x: str(x).split())

        model = Word2Vec(sentences=tokenized_column, vector_size=vector_size,
                         window=window, min_count=min_count)

        df[column] = tokenized_column.apply(lambda tokens: model.wv[tokens].mean(axis=0) if tokens else [0] * vector_size)

    df = calc(df, columns)

    return df

def calc(df, columns):

  for i in columns:

    column_mean = i + "mean"
    column_max = i + "max"
    column_min =  i + "min"

    df[column_mean] = df[i].apply(np.mean)
    df[column_max] = df[i].apply(np.max)
    df[column_min] = df[i].apply(np.min)

  return df


def main():
    
    args = create_arg_parser()

    df = read_write_data(args.data)

    if args.hashtags:
      columns = ["tweet"]
      #df = create_word2vec(df, columns)
      #df = df.drop(['tweet', 'non_text'], axis=1)
      path = 'data/' + args.data[:-4] + '_non_text.csv'
      df.to_csv(path, index=False)
    else:
      df['tweet'] = df['tweet'].apply(get_text)
      columns = ["tweet"]
      #df = create_word2vec(df, columns)
      #df = df.drop(['tweet'], axis=1)
      path = 'data/' + args.data[:-4] + '.csv'
      df.to_csv(path, index=False)



if __name__ == "__main__":
    main()