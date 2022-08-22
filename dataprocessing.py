import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

from nltk.corpus import wordnet as wn
from gensim.models import Word2Vec

import tables as tb
import torch


without_pytorch = True  # remove

check_nonzero = True  # remove

def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` from the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]

def dropcols_fancy(M, idx_to_drop):
    idx_to_drop = np.unique(idx_to_drop)
    keep = ~np.in1d(np.arange(M.shape[1]), idx_to_drop, assume_unique=True)
    return M[:, np.where(keep)[0]]

""" COMPUTE INPUT MATRICES """
# TF Matrix
def compute_tfidf_matrix(params):

	vocab = {}
	with open(params.dataset_vocab) as vocab_file:
		vocab = {word : int(idx) for line in vocab_file for (word, idx) in (line.strip().split(), ) }

	corpus = pd.read_csv(params.dataset_text, sep = '\t', header = None)

	sentences = corpus.loc[:, 2]

	vectorizer = TfidfVectorizer(use_idf = params.use_idf, vocabulary = vocab)
	V = vectorizer.fit_transform(corpus.loc[:, 2])
	print('Finished constructing TFIDF matrix')

	words = vectorizer.get_feature_names()

	V = V.tocsr()
	return V, sentences, words


# compute cosine similarity between vectors using wordnet functions
def compute_similarity(params, synset_pair):
	synsets1, synsets2 = synset_pair
	similarities_vector = [params.similarity_function(syn1, syn2) for syn1 in synsets1 for syn2 in synsets2]
	try:
		max_sim = max(similarities_vector)
	except:
		max_sim = 0
	return 0 if math.isnan(max_sim) or max_sim <= params.similarity_threshold  else max_sim


def cosine_similarity_matrix(params, embeddings, matrix_type):
	k = params.n_similar		# top k similarity values to keep - change this!
	size = len(embeddings)
	print('computed embedding shape is ', len(embeddings))

	embeddings = torch.Tensor(embeddings).to(params.device)
	embeddings_norm = embeddings / embeddings.norm(dim = 1)[:, None]
	similarity_matrix = sparse.lil_matrix((size, size))
	for i in range(0, size, params.incr):
		end = min(i+params.incr, size)
		print('On columns {} -- {}'.format(i, end))

		S_part = torch.mm(embeddings_norm, embeddings_norm[i:end].transpose(0,1))

		topk, indices = torch.topk(S_part, k, dim = 0)
		S_part_sparse = torch.zeros(S_part.shape).to(params.device)
		S_part_sparse = S_part_sparse.scatter(0, indices, topk)
		try:
			simlarity_matrix[:, i:end] = S_part_sparse.data.numpy()
		except:
			similarity_matrix[:, i:end] = S_part_sparse.cpu().data.numpy()

	similarity_matrix = similarity_matrix.transpose()
	similarity_matrix = similarity_matrix.tocsr()
	similarity_matrix.eliminate_zeros()

	# save sparse matrix
	sparse.save_npz(matrix_type + '.npz', similarity_matrix)
	return similarity_matrix


def compute_Se_cosine(params, dataset):
	# load embedding vectors
	line_count = 0
	entity_embeddings = []
	with open(params.dataset_transe, 'r') as fin:
		for line in fin:
			embedding = line.split('\t')
			assert embedding[-1] == '\n', 'ONE LINE DOES NOT HAVE ENDOFLINE'
			entity_embeddings.append([float(elem) for elem in embedding[:-1]])
			line_count += 1
	fin.close()
	entity_embeddings = np.array(entity_embeddings)
	print('entity embeddings shape - ', entity_embeddings.shape)
	np.save('EntityEmb.npy', entity_embeddings)
	return cosine_similarity_matrix(params, entity_embeddings, 'Se_'+dataset), entity_embeddings

'''
def construct_Sw_cosine(params, word_embeddings, words):
	embedding_len = len(word_embeddings)
	embeddings = torch.Tensor(word_embeddings).to(params.device)
	embeddings_norm = embeddings / embeddings.norm(dim = 1)[:, None]

	Sw = sparse.lil_matrix((embedding_len, embedding_len))
	for i in range(0, embedding_len, params.incr):
		end = min(i+params.incr, embedding_len)
		print('On columns {} -- {}'.format(i, end))

		S_part = torch.mm(embeddings_norm, embeddings_norm[i:end].transpose(0,1))
		# sparsify
		topk, indices = torch.topk(S_part, 20, dim = 0)
		S_part_sparse = torch.zeros(S_part.shape).to(params.device)
		S_part_sparse = S_part_sparse.scatter(0, indices, topk)
		try:
			Sw[:, i:end] = S_part_sparse.data.numpy()
		except:
			Sw[:, i:end] = S_part_sparse.cpu().data.numpy()

	assert Sw.nnz == 20*embedding_len, 'Odd number of nonzero values; got {} instead of {}'.format(Sw.nnz, 20*embedding_len)
	return Sw.tocsr()
'''

def compute_Sw_cosine(params, tfidf_matrix, documents, words, dataset):

	model = params.emb_model([[word for word in sentence.split()] for sentence in documents], size = 100, min_count = 0)
	embedding_len = len(list(model.wv.vocab.values()))
	vocab = list(model.wv.vocab.keys())

	missing_words = set(words) - set(vocab)

	to_drop = [i for i, j in enumerate(words) if j in missing_words]
	words = [w for w in words if w not in missing_words]   # remove missing words from list of words

	tfidf_matrix = dropcols_fancy(tfidf_matrix, to_drop)

	# compute similarity matrix
	embeddings = [model.wv[word] for word in words]
	print('Constructed embeddings vector')

	n_rows = len(embeddings)

	Sw = cosine_similarity_matrix(params, embeddings, 'Sw_'+dataset)   # TODO: CHANGE THIS - REMOVE THIS FUNCTION

	sparse.save_npz('V_'+dataset+'.npz', tfidf_matrix)
	embeddings = np.array(embeddings)
	np.save('WordEmb.npy', embeddings)
	print('word embeddings shape is ', embeddings.shape)
	return tfidf_matrix, Sw, embeddings


def compute_wordnet_similarity_words(params, words):
	all_synsets = [wn.synsets(word) for word in words]
	synset_combinations = itertools.combinations(all_synsets, 2)
	similarities_vector = [compute_similarity(params, word_pair) for word_pair in synset_combinations]
	return sparse.csr_matrix(similarities_vector)


