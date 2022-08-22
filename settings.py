""" Project settings """
from nltk.corpus import wordnet as wn
from gensim.models import FastText
from gensim.models import Word2Vec

from tempfile import mkdtemp
import os.path as path
import torch.cuda as tc
import torch


 

class Settings:
	def __init__(self):
		self.test = False
		'''		
		# Dataset
		self.dataset_text 		= '../datasets/Freebase15k/entityWords.txt'
		#self.dataset_entities 	= '../datasets/Freebase15k/train.txt'
		self.dataset_vocab 		= '../datasets/Freebase15k/word2id.txt'

	
		# Word similarity matrix
		self.use_idf        = True
		self.similarity     = 'cos'
		self.similarity_threshold = 0.8

		self.similarity_matrix_filename = path.join(mkdtemp(), 'similarity_matrix.dat')

		self.similarity_function = wn.path_similarity

		self.h5filename = 'similarity_matrix_word2vec.h5'
		self.h5omega	= 'omega_matrix_word2vec.h5'
		

		# entity embeddings settings
		self.transe_method = 'bern'   # bern or unif
		self.dataset_transe = '../datasets/entity2vec.' + self.transe_method

		# PYTORCH settings
		self.device = None

		# TEMPORARY SETTINGS:
		#self.use_pytorch_entities = True
		


		# NMF learning parameters
		self.lr = .001
		self.n_epochs = 250
		self.floatType = tc.FloatTensor # if torch.cuda.is_available else torch.FloatTensor
		self.longType = tc.LongTensor

		self.n_batches = 100
		self.n_negative = 3			# number of negative samples for each positive datasample

		self.opt = 'sgd'

		# MODEL settings
		self.n_features = 50
		'''


	def set(self, args):
		# set up input filenames
		if args.dataset == 'test':
			self.dataset_text 		= '../datasets/Test/entityWords.txt'
			self.dataset_vocab 		= '../datasets/Test/word2id.txt'
			self.dataset_transe 	= '../datasets/Test/entity2vec.' + args.transe_method	
			self.test 			= True	
			#self.n_batches = 10
		elif args.dataset == 'freebase':
			self.dataset_text 	= '../datasets/Freebase15k/entityWords.txt'
			self.dataset_vocab 		= '../datasets/Freebase15k/word2id.txt'
			self.dataset_entities = '../datasets/Freebase15k/train.txt'
			self.dataset_transe 	= '../datasets/Freebase15k/entity2vec.' + args.transe_method

		# set up device
		use_gpu = True if (args.device == 'default' and torch.cuda.is_available \
							or args.device == 'gpu') else False

		if use_gpu:
			self.device = torch.device('cuda:0')
			self.floatType = torch.cuda.FloatTensor
			self.longType = torch.cuda.LongTensor
		else:
			self.device = torch.device('cpu')
			self.floatType = torch.FloatTensor
			self.longType = torch.LongTensor

		# set up word embedding model
		if args.word_model == 'word2vec':
			self.emb_model = Word2Vec 
		else: 
			self.emb_model = FastText

		self.incr = args.incr
		self.similarity = args.similarity
		self.transe_method = args.transe_method
		self.n_similar = args.n_similar
		self.use_idf        = args.use_idf


