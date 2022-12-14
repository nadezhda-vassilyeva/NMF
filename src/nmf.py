from collections import Counter
import dataprocessing as dp
import imp
from sklearn import decomposition

import numpy as np
from scipy import sparse

import tables as tb
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.cuda as tc

from sklearn.utils import shuffle
import timeit
import matplotlib.pyplot as plt



print_time = False
print_time_similarity = False


''' MODEL '''

class NMF(torch.nn.Module):
	def __init__(self, n_entities, n_words, n_features, initE, initW, params):
		super().__init__()
		#W_data = Variable(torch.FloatTensor(n_features, n_words))
		#E_data = Variable(torch.FloatTensor(n_entities, n_features))
		W_data = Variable(torch.from_numpy(initW).type(torch.FloatTensor))
		E_data = Variable(torch.FloatTensor(initE))

		#W_data.uniform_(0, 1).abs_()
		#E_data.uniform_(0, 1).abs_()
		self.W = nn.Parameter(W_data)
		self.E = nn.Parameter(E_data)
		self.device = params.device
		self.floatType = params.floatType
		self.longType = params.longType


	def forward(self, batch):
		'''   For every (i,j) in batch, compute E[i, :]W[:, j]   '''
		row_indices = torch.LongTensor([b[0] for b in batch]).to(self.device)
		col_indices = torch.LongTensor([b[1] for b in batch]).to(self.device)
		E_sample = self.E.to(self.device).index_select(0, row_indices)
		W_sample = self.W.to(self.device).index_select(1, col_indices)

		result = (E_sample * W_sample.transpose(0,1)).sum(1).cpu()
		return result


	def positivise(self):
		''' Make W and H non-negative '''
		self.W.data = torch.clamp(self.W, min = 0.00001)
		self.E.data = torch.clamp(self.E, min = 0.00001)

	def print_params(self):
		print("W: ")
		for row in self.W:
			print(row.data)
		print()
		print('E')
		for row in self.E:
			print(row.data)

class CustomLoss(torch.nn.Module):
	# TODO: add extra term to the loss function
	def __init__(self, device):
		super(CustomLoss, self).__init__()
		self.device = device

	def penalize(self, A):
		tensor_type = tc.FloatTensor if torch.cuda.is_available else torch.FloatTensor
		return (A>0).type(tensor_type)*torch.clamp(A, max=0.)




	def entity_penalty2(self, P, S, batch, device, entity = True):
		ind = 0 if entity else 1
		S_sample_indices = torch.LongTensor([S.getrow(item[ind]).nonzero()[1] for item in batch]).to(device)
		penalty = 0

		for i, js in zip([item[ind] for item in batch], S_sample_indices):
			Pi = P.index_select(ind, torch.LongTensor([i])).to(device)
			Pj = P.to(device).index_select(ind, js)

			if not entity:
				Pi = Pi.transpose(0,1)
				Pj = Pj.transpose(0,1)


			norms = torch.norm(Pi - Pj, p = 2, dim = 1)
			S_row_i = torch.FloatTensor(S.getrow(i).toarray()[0]).to(device)
			S_sample = torch.gather(S_row_i, 0, js).to(device)
			penalty += (norms * S_sample).sum()

		return penalty

	def similarity_matrix_penalty(self, P, S, batch, device, entity = True):
		''' For entity term:
			For every i (= item[0]), find set of js (= nonzero values in row i in Se)
				For every j:
					total += Se(i,j)*(E(i) - E(j)).norm(2)
		'''
		# For every entity in the batch - find a set of entities that are similar to it
		ind = 0 if entity else 1
		i_indices_numpy = np.array([item[ind] for item in batch])
		S_sample_indices = S.tocsc()[i_indices_numpy, :].nonzero()[1]

		i_indices = torch.LongTensor(i_indices_numpy).to(device)
		n_repeat = len(S_sample_indices) // len(batch)

		Pi = P.to(device).index_select(ind, i_indices)
		j_indices_long = torch.LongTensor(S_sample_indices).to(device)
		Pj = P.to(device).index_select(ind, j_indices_long)

		if not entity:
			Pi = Pi.transpose(0,1)
			Pj = Pj.transpose(0,1)
		Pi = Pi.repeat(1, n_repeat).view(-1, Pj.shape[1])

		norms = torch.norm(Pi - Pj, p = 2, dim = 1)
		rows = i_indices_numpy.repeat(n_repeat)

		S_sample = torch.FloatTensor(S[rows, S_sample_indices]).flatten().to(device)
		penalty = (norms * S_sample).sum()
		return penalty


	def forward(self, target, prediction, reg, batch, model, Sw, Se):
		tensor_type = model.type
		tensor_type = tc.FloatTensor
		penalty = tensor_type([0])

		for p in model.parameters():
			penalty += ((p > 0).type(tc.FloatTensor)*torch.clamp(p, min = 0.).type(tc.FloatTensor)).norm(2)
		entity_penalty = self.similarity_matrix_penalty(model.E, Se, batch, model.device, True)
		word_penalty = self.similarity_matrix_penalty(model.W, Sw, batch, model.device, False)

		difference = (target.to(model.device) - prediction.to(model.device)).norm(2)
		loss = difference + \
			    reg[0]*entity_penalty + reg[1]*word_penalty + reg[2]*penalty.to(model.device)
		return loss



def rejection_sampling(batch_counter, n_negatives, high, nonzeros, already_sampled, from_entities):
	total_negative_sample = set()		# negative sample for the whole batch
	for entity, n_occurences in batch_counter.items():
		size = n_occurences * n_negatives    # size of negative sample
		# do the actual sampling
		negative_sample_for_element = set()     # keeps track of negative sample for this element in batch
		while len(negative_sample_for_element) < size:
			n_items = size - len(negative_sample_for_element)	# number of negative samples we should try getting
			random_sample = set(np.random.randint(low = 0, high = high, size = n_items))	# sample words (columns)
			if from_entities:
				random_sample = set([(entity, rs) for rs in random_sample]) - nonzeros - already_sampled   # create tuples, remove all nonzero elements
			else:
				random_sample = set([(rs, entity) for rs in random_sample]) - nonzeros - already_sampled   # same, but for words
			negative_sample_for_element |= random_sample
		total_negative_sample |= negative_sample_for_element    # add constructed negative sample to the set of negative samples for this batch
	return total_negative_sample




def plot_results(losses, differences, n_epochs, model, title):
	# plot loss and difference
	plt.plot(list(range(0, n_epochs, 50)) + [n_epochs - 1], differences, label = '|V - EW|')
	plt.xlabel("Epoch"); plt.ylabel("Error")
	plt.title(title[:-4])

	plt.plot(losses, label = 'Loss')
	plt.xlabel("Epoch")
	plt.legend()
	plt.show()

	plt.clf()
	plt.plot(losses)
	plt.title('Loss')
	plt.xlabel("Epoch"); plt.ylabel('Loss')
	plt.show()

	plt.clf()
	plt.plot(list(range(0, n_epochs, 50)) + [n_epochs - 1], differences)
	plt.title('Difference ||V - EW||')
	plt.xlabel("Epoch"); plt.ylabel('Difference')
	plt.show()



def construct_batches(n_batches, n_negatives, V, from_entities):
	row_inds, col_inds = V.nonzero()

	batchsize = np.int_(np.ceil(len(row_inds) / n_batches))

	row_inds, col_inds = shuffle(row_inds, col_inds)  # reshuffle both arrays

	nonzeros = [(i, j) for i, j in zip(row_inds, col_inds)]

	# construct positive samples
	batches = [nonzeros[i : i+batchsize] for i in range(0, len(row_inds), batchsize)]
	assert len(batches) == n_batches, "Wrong number of batches computed!"

	#print('Constructed positive samples')

	# add negative samples
	already_sampled = set()
	nonzeros = set(nonzeros)
	bcount = 0

	if from_entities: # for every (i,j) find (i,*) not in nonzeros
		for batch in batches:
			batch_counter = Counter([b[0] for b in batch])
			negative_samples = rejection_sampling(batch_counter, n_negatives, V.shape[1], nonzeros, already_sampled, from_entities)
			already_sampled |= negative_samples
			batch.extend(list(negative_samples))

	else:   # sample from entites, i.e., keep column value same, change rows
		for batch in batches:
			batch_counter = Counter([b[1] for b in batch])
			negative_samples = rejection_sampling(batch_counter, n_negatives, V.shape[0], nonzeros, already_sampled, from_entities)
			already_sampled |= negative_samples
			batch.extend(list(negative_samples))

	return batches



def get_batch_indices(V, Sw, Se, row_ind, device, n_entities):
	n = 20   	# number positive samples to pick
	nonzeros = V.indices[V.indptr[row_ind] : V.indptr[row_ind + 1]]
	zeros = np.setxor1d(nonzeros, np.arange(n_entities))
	nonzero_sample = torch.LongTensor(np.random.choice(nonzeros, size = min(n, len(nonzeros)), replace = False))
	zero_sample = torch.LongTensor(np.random.choice(zeros, size = min(5*n, len(zeros)), replace = False))
	V_samples_indices = torch.cat((nonzero_sample, zero_sample), 0)
	#V_sample = torch.gather(V[row_ind], 0, V_samples_indices)

	# get indices for Sw and Se
	# for word to word similarity matrix, we keep the same j'th (V_sample_indices), and we find all the words they are similar to - i.e. all nonzero values for each j
	word_i_indices = [Sw.indices[Sw.indptr[j]  : Sw.indptr[j+1]] for j in V_samples_indices.data]

	entity_j_indices = Se.indices[Se.indptr[row_ind] : Se.indptr[row_ind + 1]]
	return V_samples_indices, torch.LongTensor(word_i_indices), torch.LongTensor(entity_j_indices)

def get_target_values(batch, V):
	row_indices = [i[0] for i in batch]
	col_indices = [i[1] for i in batch]
	return torch.FloatTensor(V[row_indices, col_indices])
	#return torch.cat([V[i].unsqueeze(0) for i in batch])

def check_densities(V):
	rows, cols = V.nonzero()
	row_counter, col_counter = Counter(rows), Counter(cols)
	row_max = max(list(row_counter.values()))
	col_max = max(list(col_counter.values()))

	return True if row_max/V.shape[1] <= col_max/V.shape[0] else False

def init_values(V, n_features):
	mdl = decomposition.NMF(n_components = n_features)
	initE = mdl.fit_transform(V)
	initW = mdl.components_
	return initE, initW

def custom_nmf(V, Sw, Se, params, hyperparams, title, initE, initW):
	print('Without transfer learning')
	print('with momentum and lr scheduler')

	# decide where to draw negative samples from - entities (= for every (i,j) positive, find (i, *) neg) or words
	'''density_column = np.max(V.sum(0).A1)/V.shape[0]
	density_row = np.max(V.sum(1).A1)/V.shape[1]
	from_entities = True if density_row <= density_column else False
	'''

	from_entities = check_densities(V)

	n_words = Sw.shape[0]
	n_entities = Se.shape[0]

	losses = []

	# define model
	'''
	initE, initW = init_values(V, hyperparams.n_features)
	np.save('initE.npy', initE)
	np.save('initW.npy', initW)
	'''
	initW = np.load('initW.npy')
	initE = np.load('initE.npy')

	initW = np.abs(initW)
	initE = np.abs(initE)
	'''
	initW = np.random.uniform(0, 10, size = (n_words, hyperparams.n_features))
	initE = np.random.uniform(0, 10, size = (n_entities, hyperparams.n_features))
	'''
	model = NMF(n_entities, n_words, hyperparams.n_features, initE, initW, params)
	initW = None; initE = None
	print('Created NMF model')

	loss = CustomLoss(params.device)
	optimizer = hyperparams.optim(model.parameters(), **hyperparams.optim_settings)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = .1)


	# construct tensors
	V_tensor = torch.Tensor(V.todense().astype(np.float32))
	V_var = Variable(V_tensor, requires_grad = False)

	losses = []
	differences = []
	for epoch in range(hyperparams.n_epochs):
		# construct batch = one row, n positive samples, 3n negative samples
		total_loss = 0
		batches = construct_batches(hyperparams.n_batches, hyperparams.n_negatives, V, from_entities)
		bcount = -1
		for batch in batches:
			bcount += 1
			optimizer.zero_grad()

			target = get_target_values(batch, V)


			if print_time: t_start = timeit.default_timer()
			prediction = model(batch)

			l = loss(target, prediction, hyperparams.lambdas, batch, model, Sw, Se)


			l.backward(retain_graph = True)

			optimizer.step()
			model.positivise()
			total_loss += l.item()

		# test the model after each epoch
		#total_loss /= hyperparams.n_batches
		losses.append(total_loss)
		print('Epoch - {}: loss - {}'.format(epoch, total_loss))
		if epoch % 50 == 0 or epoch == hyperparams.n_epochs - 1:
			t_start = timeit.default_timer()
			V_prediction = torch.mm(model.E, model.W)
			diff = (V_tensor - V_prediction).norm(2).item()
			differences.append(diff)
			print('\t\t \t \t difference - {}'.format(diff))

		scheduler.step()
	plot_results(losses, differences, hyperparams.n_epochs, model, title)

	return model.E, model.W



