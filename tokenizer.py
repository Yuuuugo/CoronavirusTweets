from transformers import BertTokenizerFast
from transformers import TFBertModel
from transformers import RobertaTokenizerFast
from transformers import TFRobertaModel
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from transformers import BertTokenizerFast


def MLtokenizer(X_train, X_val):
    clf = CountVectorizer(max_features=128)
    X_train = clf.fit_transform(X_train)
    X_val = clf.transform(X_val)

    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train)
    X_train = tf_transformer.transform(X_train)
    X_val = tf_transformer.transform(X_val)

    return X_train, X_val


MAX_LEN = 128


def BERTtokenizer(data, max_len=MAX_LEN):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    input_ids = []
    attention_masks = []
    for i in range(len(data)):
        encoded = tokenizer.encode_plus(
            data[i],
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_attention_mask=True,
        )
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])
    return np.array(input_ids), np.array(attention_masks)
