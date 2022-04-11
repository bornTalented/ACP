import numpy as np
import tensorflow as tf
import os

from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors

class Word2Vec_GloVe_prepocess():
    def __init__(self, model_type, project_dir):

        if model_type == 'glove':
        
            glove_file = os.path.join(project_dir, 'Embeddings', 'glove.6B', 'glove.6B.300d.txt.word2vec')
            # Load vectors directly from the file
            self.wv_embeddings = KeyedVectors.load_word2vec_format(glove_file, binary=False)
        
        elif model_type == 'word2vec':

            word2vec_file = os.path.join(project_dir, 'Embeddings', 'Word2Vec', 'GoogleNews-vectors-negative300.bin.gz')
            # Load vectors directly from the file
            self.wv_embeddings = KeyedVectors.load_word2vec_format(word2vec_file, binary=True)

        

    def generate_Word2Vec_GloVe_emb(self, data_frame,  MAX_SEQUENCE_LENGTH):

        tokens_input = data_frame.review.apply((lambda x: word_tokenize(x)))
        tokens_length = tokens_input.apply((lambda x: len(x)))

        if not MAX_SEQUENCE_LENGTH:
            MAX_SEQUENCE_LENGTH = max(tokens_length)

        tokens_input_padded = tf.keras.preprocessing.sequence.pad_sequences(tokens_input, value='', dtype=object,
                                                                            maxlen=MAX_SEQUENCE_LENGTH, padding='post')

        emb = self.get_emb(tokens_input_padded)

        return emb, MAX_SEQUENCE_LENGTH
    


    def get_emb(self, corpus):
        emb = []
        EMB_DIM = self.wv_embeddings['apple'].size
        for sentence in corpus:
            e = []
            for x in sentence:
                try:
                    e.append(self.wv_embeddings[x])
                except KeyError:  # getting OOV emb as allzeros vector
                    e.append(np.zeros(EMB_DIM))
            emb.append(np.array(e))
        return np.array(emb)
