import re
import argparse
import random
import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader

np.random.seed(42)
random.seed(42)


class TweetDataset(Dataset):
    '''Custom dataset for handling tweet data and labels for RoBERTa.'''
    def __init__(self, tweets, labels, tokenizer, max_length=128):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = str(self.tweets[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", action='store_true', help="Predict on the test files.")
    args = parser.parse_args()
    return args


def compute_metrics(pred):
    '''Compute metrics for evaluation.'''
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='binary', pos_label=1)
    recall = recall_score(labels, preds, average='binary', pos_label=1)
    f1 = f1_score(labels, preds, average='binary', pos_label=1)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    args = create_arg_parser()

    # Load data
    train = pd.read_csv("data/train.csv")
    dev = pd.read_csv("data/dev.csv")
    test = pd.read_csv("data/test.csv")

    # Preprocess text data
    train_texts = train['tweet'].apply(lambda x: x if isinstance(x, str) else "")
    dev_texts = dev['tweet'].apply(lambda x: x if isinstance(x, str) else "")
    test_texts = test['tweet'].apply(lambda x: x if isinstance(x, str) else "")

    # Label encoding
    label_map = {"NOT": 0, "OFF": 1}
    y_train = train["label"].map(label_map).values
    y_dev = dev["label"].map(label_map).values
    y_test = test["label"].map(label_map).values

    # Load RoBERTa tokenizer and model with classification head
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    # Prepare datasets
    train_dataset = TweetDataset(train_texts.tolist(), y_train, tokenizer)
    dev_dataset = TweetDataset(dev_texts.tolist(), y_dev, tokenizer)
    test_dataset = TweetDataset(test_texts.tolist(), y_test, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch"
    )

    # Set up Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    print("Training RoBERTa...")
    trainer.train()

    # Evaluate on dev set
    print("\nEvaluating on development set:")
    dev_metrics = trainer.evaluate()
    print(dev_metrics)

    # Evaluate on test set if specified
    if args.test:
        print("\nEvaluating on test set:")
        test_metrics = trainer.evaluate(test_dataset)
        print(test_metrics)


if __name__ == "__main__":
    main()
