

'''
Hyperparams settings for NMF
'''
import torch

class Hyperparameters:
	def __init__(self):
		self.lambdas = [.01, .01, .2]  # regularizer params for Se, Sw, and model parameters
		self.momentum = .4

	def set(self, args):
		self.n_epochs = args.n_epochs 
		self.lr = args.lr
		self.n_batches = args.n_batches
		self.n_features = args.n_features
		self.n_negatives = args.n_negatives


		if args.optim == 'sgd':
			self.optim = torch.optim.SGD 
			self.optim_settings = {'lr': self.lr, 'momentum': self.momentum}

		elif args.optim == 'adam':
			self.optim = torch.optim.Adam
			wd = .001
			self.optim_settings.update({'weight_decay:', wd})
		else: 		# adagrad
			self.optim = torch.optim.Adagrad
			lr_decay = .001
			self.optim_settings.update({'lr_decay': lr_decay})

	def print_hp(self):
		settings = ("Optimizer: {}, l_rate: {}, # epochs: {}," \
			" # batches: {}".format(self.optim.__name__, \
				self.lr, self.n_epochs, self.n_batches ))
		print(settings)

	def print_lambdas(self):
		lambdas_string = ('Regularization params: Se - {}, ' \
						  'Sw - {}, P - {}'.format(*self.lambdas))
		print(lambdas_string)
