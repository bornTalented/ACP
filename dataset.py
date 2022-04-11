import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

class SemEval_Dataset():

	def __init__(self, opt, for_aspect_category):

		
		self.for_aspect_category = for_aspect_category
		

		self.training_data_frame = self.read_semeval2014(xml_file=opt.dataset_files['train'], shuffling=True)
		self.testing_data_frame = self.read_semeval2014(xml_file=opt.dataset_files['test'], shuffling=False)



	def read_semeval2014(self, xml_file, shuffling=False):

	    tree = ET.parse(xml_file)
	    root = tree.getroot()
	    # data = [{'b': 2, 'c':3}, {'a': 10, 'b': 20, 'c': 30}]
	    data = []
	    for k, sentence in enumerate(root.findall("./sentence")):

	        _id = sentence.get('id')
	        text = sentence.find('text').text.lower().strip()

	        if self.for_aspect_category:
	            for aspectCategory in sentence.findall("./aspectCategories/aspectCategory"):
	                aspect_category = aspectCategory.get('category').lower()
	                aspect_polarity = aspectCategory.get('polarity')

	                my_dict = {}
	                my_dict['id'] = _id
	                my_dict['review'] = text
	                my_dict['aspect_category'] = aspect_category
	                my_dict['aspect_category_polarity'] = aspect_polarity
	                data.append(my_dict)
	        else:
	            for aspectTerm in sentence.findall("./aspectTerms/aspectTerm"):
	                aspect_term = aspectTerm.get('term').lower()
	                aspect_polarity = aspectTerm.get('polarity')

	                from_ = int(aspectTerm.get('from'))
	                to_ = int(aspectTerm.get('to'))

	                my_dict = {}
	                my_dict['id'] = _id
	                my_dict['review'] = text
	                my_dict['aspect_term'] = aspect_term
	                my_dict['aspect_term_polarity'] = aspect_polarity
	                my_dict['from'] = from_
	                my_dict['to'] = to_
	                data.append(my_dict)
	    df = pd.DataFrame(data).dropna()  # droping rows having NAN values

	    if shuffling:
	        df.sample(frac=1).reset_index(drop=True)

	    return df


	# # from loadDataset import tag_aspect_terms
	# def is_aspect_present(l, a, frm, to):
	#     '''
	#     l is the list of sentence tokens
	#     a is the list of aspect-terms tokens

	#     mask_list contains True corresponding to aspect-terms tokens
	#     '''

	#     present_value = 1
	#     absent_value = 0

	#     l = np.array(l)
	#     a = np.array(a)
	#     is_present = False
	#     mask_list = np.repeat(absent_value, len(l))

	#     # find the index of first aspect-term token
	#     ind = np.where(l == a[0])
	#     for i in ind[0]:
	#         aspect_ind = np.arange(len(a)) + i

	#         if aspect_ind[-1] < len(l):  # check for out-of-index
	#             if np.array_equal(l[aspect_ind], a):
	#                 is_present = True
	#                 mask_list[aspect_ind] = present_value

	#     # ---------- checking for problem
	#     indices = np.nonzero(mask_list)[0]

	#     if not np.all(np.diff(indices, 1) == 1):  # indices are NOT in order one-by-one

	#         mask_list = np.repeat(absent_value, len(l))
	#         length = 0
	#         for i, t in enumerate(l):
	#             if frm <= length and length < to:
	#                 is_present = True
	#                 mask_list[i] = present_value
	#             length += len(t) + 1  # plus 1 for SPACE
	#     # ----------

	#     return is_present, mask_list


	# def replace_tokens(sentence_tokens, aspect_tokens):
	#     sentence_tokens = np.array(sentence_tokens)
	#     # check char by char [this code will replace the sentence token by aspect token]

	#     indices = []
	#     for a in aspect_tokens:
	#         for k, t in enumerate(sentence_tokens):
	#             if t.find(a) != -1:
	#                 indices.append(k)
	#     if np.all(np.diff(indices, 1) == 1):  # indices are in order one-by-one
	#         sentence_tokens[indices] = aspect_tokens
	#     else:
	#         print("Unable to process following sentence tokens:")
	#         print('aspect tokens:', aspect_tokens)
	#         print('review tokens:', sentence_tokens)

	#     return sentence_tokens


	# def tag_aspect_terms(df_aspect_list, df_token_list, spacy_model, df_from, df_to):
	#     '''
	#     this function finds the data-frame indices which are unable to find aspect-terms tokens in the review
	#     due to some inconsistency
	#     it return the indices for such reviews and
	#     also return the pandas series containing masked corresponding to aspect-terms in token-list
	#     '''
	#     # extracting indices which do not have aspect-terms in review token-list
	#     pd_ind = []
	#     mask_aspects_list = []

	#     present_value = 1  # can replace 1 by True
	#     absent_value = 0  # can replace 0 by False

	#     for i in df_token_list.index:

	#         sentence_tokens = df_token_list[i]
	#         aspect_tokens = df_aspect_list[i]

	#         is_present, mask_list = is_aspect_present(sentence_tokens, aspect_tokens, df_from[i], df_to[i])

	#         if not is_present:  # when aspect-term is not tokenized properly
	#             print('old aspect tokens:', aspect_tokens)
	#             print('old review tokens:', sentence_tokens)

	#             sentence_tokens = replace_tokens(sentence_tokens, aspect_tokens)
	#             df_token_list[i] = list(sentence_tokens)
	#             print('new review tokens:', sentence_tokens)
	#             print()

	#             is_present, mask_list = is_aspect_present(sentence_tokens, aspect_tokens, df_from[i], df_to[i])

	#             pd_ind.append(i)

	#         mask_aspects_list.append(mask_list)

	#     print("Updated tokens indices: ", pd_ind)
	#     #     rowData = training_data_frame.loc[pd_ind , : ]

	#     return df_token_list, pd.Series(mask_aspects_list, index=df_token_list.index)

	# # def is_aspect_present(l, a):
	# #     '''
	# #     l is the list of sentence tokens
	# #     a is the list of aspect-terms tokens
	# #
	# #     mask_list contains True corresponding to aspect-terms tokens
	# #     '''
	# #
	# #     present_value = 1
	# #     absent_value = 0
	# #
	# #     l = np.array(l)
	# #     a = np.array(a)
	# #     is_present = False
	# #     mask_list = np.repeat(absent_value, len(l))
	# #
	# #     # find the index of first aspect-term token
	# #     ind = np.where(l == a[0])
	# #     for i in ind[0]:
	# #         aspect_ind = np.arange(len(a)) + i
	# #
	# #         if aspect_ind[-1] < len(l):  # check for out-of-index
	# #             if np.array_equal(l[aspect_ind], a):
	# #                 is_present = True
	# #                 mask_list[aspect_ind] = present_value
	# #
	# #     return is_present, mask_list
	# #
	# #
	# # def tag_aspect_terms(data_frame, df_token, spacy_model):
	# #     '''
	# #     this function finds the data-frame indices which are unable to find aspect-terms tokens in the review
	# #     due to some inconsistency
	# #     it return the indices for such reviews and
	# #     also return the pandas series containing masked corresponding to aspect-terms in token-list
	# #     '''
	# #     # extracting indices which do not have aspect-terms in review token-list
	# #     ind = []
	# #     mask_aspects_list = []
	# #
	# #     present_value = 1  # can replace 1 by True
	# #     absent_value = 0  # can replace 0 by False
	# #
	# #     for i in df_token.index:
	# #
	# #         sentence_tokens = df_token[i]
	# #         aspect = data_frame.aspect_term[i]
	# #
	# #         if aspect in sentence_tokens:
	# #             mask_list = np.repeat(absent_value, len(sentence_tokens))
	# #             mask_list[sentence_tokens.index(aspect)] = present_value
	# #             mask_aspects_list.append(mask_list)
	# #         else:
	# #             # aspect may be a phrase
	# #             aspect_tokens = [token.text for token in spacy_model(aspect)]  # convert aspect-terms to tokens
	# #
	# #             is_present, mask_list = is_aspect_present(sentence_tokens, aspect_tokens)
	# #
	# #             mask_aspects_list.append(mask_list)
	# #             if not is_present:  # when aspect-term is not tokenized properly
	# #                 print('id:', data_frame.id[i])
	# #                 print('aspect-terms:', aspect)
	# #                 print('aspect tokens:', aspect_tokens)
	# #                 print('review tokens:', sentence_tokens)
	# #                 print()
	# #                 ind.append(i)
	# #
	# #     return ind, pd.Series(mask_aspects_list, index=df_token.index)