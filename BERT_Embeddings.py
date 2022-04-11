import tensorflow as tf
import tensorflow_hub as hub
import bert



class BERT_prepocess():
    def __init__(self, max_seq_length = 128):

        model_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
        self.BERTmodel = self.BERT_embedding_model(model_url, max_seq_length)

        # tf.keras.utils.plot_model(BERTmodel, show_shapes=True, show_layer_names=True
        #     # , to_file='ATE_APD_full.png'
        #     )

        self.tokenizer = self.get_BERT_tokenizer()
        
    
    def BERT_embedding_model(self, model_url, max_seq_length):
        # max_seq_length= 128 Your choice here.
        self.bert_layer = hub.KerasLayer(model_url, trainable=False)

        input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")

        pooled_output, sequence_output = self.bert_layer([input_word_ids, input_mask, segment_ids])

        model = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])
        model.summary()

        return model

    def get_BERT_tokenizer(self):

        # Import tokenizer using the original vocab file
        vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
        
        FullTokenizer = bert.bert_tokenization.FullTokenizer
        tokenizer = FullTokenizer(vocab_file, do_lower_case)

        return tokenizer

    def generate_BERT_emb(self, data_frame, max_seq_length, print_data=False):

        

        x_tokenized = self.tokenize_data_BERT(data_frame, max_seq_length)
        # input_ids = self.get_ids(stokens, max_seq_length)
        # input_masks = self.get_masks(stokens, max_seq_length)
        # input_segments = self.get_segments(stokens, max_seq_length)

        input_ids = x_tokenized.apply((lambda x: self.get_ids(x, max_seq_length)))
        input_masks = x_tokenized.apply((lambda x: self.get_masks(x, max_seq_length)))
        input_segments = x_tokenized.apply((lambda x: self.get_segments(x, max_seq_length)))

        pool_embs, all_embs = self.BERTmodel.predict([list(input_ids), list(input_masks), list(input_segments)])

        if (print_data):
            print('input_ids...')
            print(input_ids)
            print('input_masks...')
            print(input_masks)
            print('input_segments...')
            print(input_segments)

            print('pool_embs.shape:', pool_embs.shape, 'all_embs.shape:', all_embs.shape)

        return pool_embs, all_embs

    def tokenize_data_BERT(self, data_frame, max_seq_length):
        tokenized_data = data_frame.review.apply((lambda x: self.tokenizer.tokenize(x)))
        tokenized_data = tokenized_data.apply((lambda x: ["[CLS]"] + self.check_token_length(x, max_seq_length) + ["[SEP]"]))

        return tokenized_data

    def get_ids(self, tokens, max_seq_length):
        """Token ids from Tokenizer vocab"""
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
        return input_ids
        
    def get_masks(self, tokens, max_seq_length):
        """Mask for padding"""
        #     if len(tokens)>max_seq_length:
        #         raise IndexError("Token length more than max seq length!")
        return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


    def get_segments(self, tokens, max_seq_length):
        """Segments: 0 for the first sequence, 1 for the second"""

        #     if len(tokens)>max_seq_length:
        #         raise IndexError("Token length more than max seq length!")
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments + [0] * (max_seq_length - len(tokens))



    def check_token_length(self, tokens, max_seq_length):
        msl = max_seq_length - 2
        if len(tokens) > msl:
            tokens = tokens[:msl]
        return tokens


# stokens = tokenizer.tokenize(s) # for single string
# stokens = ["[CLS]"] + stokens + ["[SEP]"]















