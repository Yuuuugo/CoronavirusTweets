"""
Preprocessing steps, will return train_ds and test_ds where you can just apply 
your fitting and evaluation function
"""

import pandas as pd
import cleaning
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tokenizer import MLtokenizer, BERTtokenizer

seed = 1


def extract_dataset(df, transformers):
    ros = RandomOverSampler()
    train_x, train_y = ros.fit_resample(
        np.array(df["clean_text"]).reshape(-1, 1),
        np.array(df["Sentiment"]).reshape(-1, 1),
    )
    train_os = pd.DataFrame(
        list(zip([x[0] for x in train_x], train_y)), columns=["clean_text", "Sentiment"]
    )
    X = train_os["clean_text"].values
    y = train_os["Sentiment"].values
    onehotencoder = OneHotEncoder()
    ohe = OneHotEncoder(sparse=True)
    if transformers:
        # If we use a DeepLearning network we onehot encode the value
        y = ohe.fit_transform(np.array(y).reshape(-1, 1)).toarray()
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=seed,
    )
    # y_train = onehotencoder.fit_transform(y_train)
    if transformers:
        MAX_LEN = 128
        train_input_ids, train_attention_masks = BERTtokenizer(X_train, MAX_LEN)
        val_input_ids, val_attention_masks = BERTtokenizer(X_valid, MAX_LEN)

        return (
            train_input_ids,
            val_input_ids,
            train_attention_masks,
            val_attention_masks,
            y_train,
            y_valid,
        )
    else:
        X_train, X_valid = MLtokenizer(X_train, X_valid)
        return X_train, X_valid, y_train, y_valid


def load_data(transformers=False):
    df = pd.read_csv("data/Corona_NLP_train.csv", encoding="ISO-8859-1")
    df.rename(columns={"OriginalTweet": "text"}, inplace=True)
    df = df[["text", "Sentiment"]]
    df = cleaning.cleaning_df(df)

    if transformers:
        (
            train_input_ids,
            val_input_ids,
            train_attention_masks,
            val_attention_masks,
            y_train,
            y_valid,
        ) = extract_dataset(df, transformers)
        return (
            train_input_ids,
            val_input_ids,
            train_attention_masks,
            val_attention_masks,
            y_train,
            y_valid,
        )
    else:
        X_train, X_valid, y_train, y_valid = extract_dataset(df, transformers)
        return X_train, X_valid, y_train, y_valid


def dataset(x, y, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch()


if __name__ == "__main__":
    train_df = pd.read_csv("data/Corona_NLP_train.csv", encoding="latin-1")

    X_train, X_valid, y_train, y_valid = load_data()

    print(X_train.shape)
    print(y_train.shape)
