import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
from nltk.tokenize import word_tokenize

class ELMO_prepocess():
    def __init__(self):

        model_url = "https://tfhub.dev/google/elmo/3"
        self.ELMOmodel = self.ELMO_embedding_model(model_url)

        # tf.keras.utils.plot_model(ELMOmodel, show_shapes=True, show_layer_names=True
        #     # , to_file='ATE_APD_full.png'
        #     )
    

    def ELMO_embedding_model(self, model_url):

        self.elmo_layer = hub.KerasLayer(model_url, trainable=False, signature="tokens", output_key="elmo")

        #### See Also: https://github.com/tensorflow/hub/issues/402
        tokens = tf.keras.layers.Input(shape=[None], dtype=tf.string)
        seq_lens = tf.keras.layers.Input(shape=[], dtype=tf.int32)

        out = self.elmo_layer({"tokens": tokens, "sequence_len": seq_lens})
        model = tf.keras.Model(inputs=[tokens, seq_lens], outputs=out)
        # model.compile("adam", loss="sparse_categorical_crossentropy")
        model.summary()

        return model    
    
    def generate_ELMO_emb(self, data_frame, max_seq_length, print_data=False):

        # MAX_SEQUENCE_LENGTH is captured using training data only
        tokens_input_padded, tokens_length, MAX_SEQUENCE_LENGTH = self.tokenize_data_EMLO(data_frame, MAX_SEQUENCE_LENGTH=max_seq_length)

        if (print_data):
            print(tokens_input_padded, tokens_length, sep='\n')

        elmo_embs = self.ELMOmodel.predict([tokens_input_padded, tokens_length])

        return elmo_embs, MAX_SEQUENCE_LENGTH

    def tokenize_data_EMLO(self, data_frame, MAX_SEQUENCE_LENGTH=''):
        #     tokens_input = data_frame.review.apply((lambda x :  x.split()))

        tokens_input = data_frame.review.apply((lambda x: word_tokenize(x)))
        tokens_length = tokens_input.apply((lambda x: len(x)))

        if not MAX_SEQUENCE_LENGTH:
            MAX_SEQUENCE_LENGTH = max(tokens_length)

        tokens_input_padded = tf.keras.preprocessing.sequence.pad_sequences(tokens_input, value='', dtype=object,
                                                                            maxlen=MAX_SEQUENCE_LENGTH, padding='post')

        tokens_length = np.array(tokens_length).astype('int32')

        return tokens_input_padded, tokens_length, MAX_SEQUENCE_LENGTH



