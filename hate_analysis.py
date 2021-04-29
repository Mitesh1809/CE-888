"""
Classifying sentiments of a tweet
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import official.nlp.bert.tokenization as tokenization
import official.nlp.optimization as opti
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder

from reading_hate_tweet import get_hate_data

tf.get_logger().setLevel('ERROR')


def encode_names(tokenizer, n):
    tokens = list(tokenizer.tokenize(n))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)


def bert_encode(string_list, tokenizer, max_seq_length):
    string_tokens = tf.ragged.constant([
        encode_names(tokenizer, n) for n in np.array(string_list)])
    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * string_tokens.shape[0]
    input_word_ids = tf.concat([cls, string_tokens], axis=-1)
    input_mask = tf.ones_like(input_word_ids).to_tensor(
        shape=(None, max_seq_length))
    type_cls = tf.zeros_like(cls)
    type_tokens = tf.ones_like(string_tokens)
    input_type_ids = tf.concat([type_cls, type_tokens], axis=-1).to_tensor(shape=(None, max_seq_length))
    inputs = {
        'input_word_ids': input_word_ids.to_tensor(shape=(None, max_seq_length)),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids
    }
    return inputs


if __name__ == "__main__":
    train_data, test_data, val_data, mappings = get_hate_data()

    # Cleaning tweets.
    train_data.tweets = train_data.tweets.transform(
        lambda x: x.lower().replace("@user", "").strip())
    test_data.tweets = test_data.tweets.transform(
        lambda x: x.lower().replace("@user", "").strip())
    val_data.tweets = val_data.tweets.transform(
        lambda x: x.lower().replace("@user", "").strip())

    # Creating labels.
    label_encoder = LabelEncoder()
    train_data.labels = label_encoder.fit_transform(train_data.labels)
    test_data.labels = label_encoder.transform(test_data.labels)
    val_data.labels = label_encoder.transform(val_data.labels)

    y_train = tf.keras.utils.to_categorical(train_data.labels)   # [:10000]
    y_test = tf.keras.utils.to_categorical(test_data.labels)   # [:2000]
    y_val = tf.keras.utils.to_categorical(val_data.labels)   # [:1000]

    print("Fetching BERT model")
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2", trainable=True)
    print("Model is fetched")
    # bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/
    # bert_en_uncased_L-8_H-512_A-8/2", trainable=True)

    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
    tweets = tf.ragged.constant([encode_names(tokenizer, n) for n in train_data.tweets])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * tweets.shape[0]
    input_word_ids = tf.concat([cls, tweets], axis=1)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    type_cls = tf.zeros_like(cls)
    type_tweet = tf.ones_like(tweets)
    input_type_ids = tf.concat([type_cls, type_tweet], axis=1).to_tensor()

    max_seq_len = max([len(i) for i in input_word_ids])
    max_seq_len = int(1.5 * max_seq_len)

    # X_train = bert_encode(train_data.tweets[:10000], tokenizer, max_seq_len)
    # X_test = bert_encode(test_data.tweets[:2000], tokenizer, max_seq_len)
    # X_val = bert_encode(val_data.tweets[:1000], tokenizer, max_seq_len)
    X_train = bert_encode(train_data.tweets, tokenizer, max_seq_len)
    X_test = bert_encode(test_data.tweets, tokenizer, max_seq_len)
    X_val = bert_encode(val_data.tweets, tokenizer, max_seq_len)

    num_class = len(label_encoder.classes_)
    max_seq_length = max_seq_len

    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    output = tf.keras.layers.Dropout(rate=0.1)(pooled_output)
    output = tf.keras.layers.Dense(num_class, activation='softmax', name='output')(output)

    model = tf.keras.Model(
        inputs={
            'input_word_ids': input_word_ids,
            'input_mask': input_mask,
            'input_type_ids': segment_ids
        },
        outputs=output)

    epochs = 3
    batch_size = 4
    eval_batch_size = batch_size

    train_data_size = len(y_train)
    steps_per_epoch = int(train_data_size / batch_size)
    num_train_steps = steps_per_epoch * epochs
    warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

    optimizer = opti.create_optimizer(1e-3, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)
    model.save(os.path.join(".", "hate_model_final"))
    print(model.evaluate(X_test, y_test))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

