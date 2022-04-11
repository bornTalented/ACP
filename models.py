import tensorflow as tf
import pandas as pd
import numpy as np
import os
import math


def ACP_Model(opt):

    cat_weights_file = os.path.join('data', 'category_weights',f'category_weights_{opt.model_type}.csv')
    cat_embedding_layer = get_category_embeddings_layer(cat_weights_file, opt.category_list)

    LSTM_UNIT = math.ceil(opt.emb_dim / 2)

    input_categories = tf.keras.Input(shape=(1,), dtype='int32', name='categoryInputLayer')
    embedded_input_categories = cat_embedding_layer(input_categories)

    embedded_sequences_words = tf.keras.Input(shape=(opt.max_seq_len, opt.emb_dim), name='wordInputLayer')

    # -----------------------------------------------------------------------------------------------#
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(LSTM_UNIT, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)
    )(embedded_sequences_words)
    x = tf.keras.layers.concatenate([x, embedded_input_categories], axis=1)
    x = tf.keras.layers.Flatten()(x)
    preds = tf.keras.layers.Dense(opt.prediction_unit, activation='softmax', name='predictions')(x)
    # -----------------------------------------------------------------------------------------------#

    model = tf.keras.Model(inputs=[embedded_sequences_words, input_categories], outputs=preds)

    # Specify the training configuration (optimizer, loss, metrics)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),  # Optimizer
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # Loss function to minimize
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])  # List of metrics to monitor

    model.summary()

    return model


def ASP_Model(opt):

    cat_weights_file = os.path.join('data', 'category_weights',f'category_weights_{opt.model_type}.csv')
    cat_embedding_layer = get_category_embeddings_layer(cat_weights_file, opt.category_list)

    input_categories = tf.keras.Input(shape=(1,), dtype='int32', name='categoryInputLayer')
    embedded_input_categories = cat_embedding_layer(input_categories)


    embedded_sequences_words = tf.keras.Input(shape=(opt.max_seq_len, opt.emb_dim), name='wordInputLayer')

    # -----------------------------------------------------------------------------------------------#
    x = tf.keras.layers.UpSampling1D(opt.max_seq_len)(embedded_input_categories)
    # x = tf.keras.layers.Lambda(lambda x: repeat_elements(x, opt.max_seq_len, axis=1))(embedded_input_categories)
    x = tf.keras.layers.concatenate([embedded_sequences_words, x], axis=-1)
    x = tf.keras.layers.LSTM(opt.lstm_unit, dropout=0.5, recurrent_dropout=0.5)(x)
    preds = tf.keras.layers.Dense(opt.prediction_unit, activation='softmax', name='predictions')(x)
    # -----------------------------------------------------------------------------------------------#

    model = tf.keras.Model(inputs=[embedded_sequences_words, input_categories], outputs=preds)

    # Specify the training configuration (optimizer, loss, metrics)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),  # Optimizer
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # Loss function to minimize
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])  # List of metrics to monitor

    model.summary()

    return model
    

def binary_category_prediction_model(max_seq_len, emb_dim, UNIT):
    embedded_sequences_words = tf.keras.Input(shape=(max_seq_len, emb_dim), name='wordInputLayer')
    x = tf.keras.layers.Flatten()(embedded_sequences_words)
    x = tf.keras.layers.Dense(UNIT)(x)

    # x = tf.keras.layers.LSTM(LSTM_UNIT, dropout=0.5, recurrent_dropout=0.5)(embedded_sequences_words)
    # x = tf.keras.layers.Bidirectional(
    #         tf.keras.layers.LSTM(LSTM_UNIT)
    #                                  )(embedded_sequences_words)

    preds = tf.keras.layers.Dense(1, activation='sigmoid', name='predictions')(x)

    model = tf.keras.Model(inputs=embedded_sequences_words, outputs=preds)

    # Specify the training configuration (optimizer, loss, metrics)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),  # Optimizer
                  loss=tf.keras.losses.BinaryCrossentropy(), # Loss function to minimize
                  metrics=[tf.keras.metrics.Accuracy()]) # List of metrics to monitor

    model.summary()

    return model


def get_category_embeddings_layer(category_weights_file_name, category_list):

    category_weights_word2vec = pd.read_csv(category_weights_file_name, usecols = category_list)  
    
    max_seq_len = len(category_list)
    EMBEDDING_DIM = len(category_weights_word2vec)
    
    # embedding matrix indices are in following order: ['service', 'food', 'anecdotes/miscellaneous', 'ambience', 'price']) 
    cat_emb_matrix = np.transpose(np.array(category_weights_word2vec))

    cat_embedding_layer = tf.keras.layers.Embedding(max_seq_len, EMBEDDING_DIM, trainable = False)
    cat_embedding_layer.build((None,))
    cat_embedding_layer.set_weights([cat_emb_matrix])
    
    return cat_embedding_layer