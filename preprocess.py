import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

class Preprocessor():

	def __init__(self, model_type, polarity_index):
		self.model_type = model_type.lower()
		assert self.model_type in ['bert', 'elmo', 'word2vec', 'glove']
		
		self.polarity_index = polarity_index


	def preprocess_data(self, training_data_frame, testing_data_frame, category_dict, max_seq_length, project_dir):

		if self.model_type in ['glove', 'word2vec']:
			
			from Word2Vec_GloVe_Embeddings import Word2Vec_GloVe_prepocess
			wgp = Word2Vec_GloVe_prepocess(self.model_type, project_dir)

			embeddings_train, MAX_SEQUENCE_LENGTH = wgp.generate_Word2Vec_GloVe_emb(training_data_frame, max_seq_length)
			embeddings_test, _ = wgp.generate_Word2Vec_GloVe_emb(testing_data_frame, MAX_SEQUENCE_LENGTH)

		elif self.model_type == 'elmo':
			
			from ELMO_Embeddings import ELMO_prepocess
			ep = ELMO_prepocess()
			
			embeddings_train, MAX_SEQUENCE_LENGTH  = ep.generate_ELMO_emb(training_data_frame, max_seq_length, print_data=False)
			embeddings_test, _ = ep.generate_ELMO_emb(testing_data_frame, MAX_SEQUENCE_LENGTH, print_data=False)

		elif self.model_type == 'bert':

			from BERT_Embeddings import BERT_prepocess
			bp = BERT_prepocess(max_seq_length)

			pool_embs_train, embeddings_train = bp.generate_BERT_emb(training_data_frame, max_seq_length, print_data=False)
			pool_embs_test, embeddings_test = bp.generate_BERT_emb(testing_data_frame, max_seq_length, print_data=False)
		
		

		x_train, y_train = embeddings_train , np.array(training_data_frame['aspect_category_polarity'].replace(self.polarity_index, inplace=False))
		x_test, y_test = embeddings_test , np.array(testing_data_frame['aspect_category_polarity'].replace(self.polarity_index, inplace=False))


		# converting 'string to numeric labels'
		x_train_cat = np.array(training_data_frame['aspect_category'].replace(category_dict, inplace=False))
		x_test_cat = np.array(testing_data_frame['aspect_category'].replace(category_dict, inplace=False))
		
		return x_train, x_train_cat, y_train, x_test, x_test_cat, y_test

