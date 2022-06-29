from transformers import TFBertModel
import numpy as np
import tensorflow as tf


def create_model(max_len=128):
    bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    ##params###
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-7)
    loss = tf.keras.losses.CategoricalCrossentropy()
    accuracy = tf.keras.metrics.CategoricalAccuracy()

    input_ids = tf.keras.Input(shape=(max_len,), dtype="int32")

    attention_masks = tf.keras.Input(shape=(max_len,), dtype="int32")

    embeddings = bert_model([input_ids, attention_masks])[1]

    output = tf.keras.layers.Dense(5, activation="softmax")(embeddings)

    model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=output)

    model.compile(opt, loss=loss, metrics=accuracy)

    return model
