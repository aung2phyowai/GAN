# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import eeggan.util as utils


# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import eeggan.util as utils
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np

class GAN_Module(nn.Module):
	"""
	Parent module for different GANs

	Attributes
	----------
	optimizer : torch.optim.Optimizer
		Optimizer for training the model parameters
	loss : torch.nn.Loss
		Loss function
	"""
	def __init__(self):
		super(GAN_Module, self).__init__()

		self.did_init_train = False

	def save_model(self,fname):
		"""
		Saves `state_dict` of model and optimizer

		Parameters
		----------
		fname : str
			Filename to save
		"""
		cuda = False
		if next(self.parameters()).is_cuda: cuda = True
		cpu_model = self.cpu()
		model_state = cpu_model.state_dict()
		opt_state = cpu_model.optimizer.state_dict()

		torch.save((model_state,opt_state,self.did_init_train),fname)
		if cuda:
			self.cuda()

	def load_model(self,fname):
		"""
		Loads `state_dict` of model and optimizer

		Parameters
		----------
		fname : str
			Filename to load from
		"""
		model_state,opt_state,self.did_init_train = torch.load(fname)

		self.load_state_dict(model_state)
		self.optimizer.load_state_dict(opt_state)



class WGAN_I_Discriminator(GAN_Module):
	"""
	Improved Wasserstein GAN discriminator

	References
	----------
	Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017).
	Improved Training of Wasserstein GANs.
	Retrieved from http://arxiv.org/abs/1704.00028
	"""
	def __init__(self):
		super(WGAN_I_Discriminator, self).__init__()

	def train_init(self,alpha=1e-4,betas=(0.5,0.9),
				   lambd=10,one_sided_penalty=False,distance_weighting=False,
				   eps_drift=0.,eps_center=0.,lambd_consistency_term=0.):
		"""
		Initialize Adam optimizer for discriminator

		Parameters
		----------
		alpha : float, optional
			Learning rate for Adam
		betas : (float,float), optional
			Betas for Adam
		lambda : float, optional
			Weight for gradient penalty (default: 10)
		one_sided_penalty : bool, optional
			Use one- or two-sided penalty
			See Hartmann et al., 2018 (default: False)
		distance_weighting : bool
			Use distance-weighting
			See Hartmann et al., 2018 (default: False)
		eps_drift : float, optional
			Weigth to keep discriminator output from drifting away from 0
			See Karras et al., 2017 (default: 0.)
		eps_center : float, optional
			Weight to keep discriminator centered at 0
			See Hartmann et al., 2018 (default: 0.)

		References
		----------
		Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
		Progressive Growing of GANs for Improved Quality, Stability,
		and Variation. Retrieved from http://arxiv.org/abs/1710.10196
		Hartmann, K. G., Schirrmeister, R. T., & Ball, T. (2018).
		EEG-GAN: Generative adversarial networks for electroencephalograhic
		(EEG) brain signals. Retrieved from https://arxiv.org/abs/1806.01875
		"""
		# super(WGAN_I_Discriminator,self).train_init(alpha,betas)

		self.optimizer = optim.Adam(self.parameters(),lr=alpha,betas=betas)
		self.loss = torch.nn.BCELoss()
		self.did_init_train = True

		self.loss = None
		self.lambd = lambd
		self.one_sided_penalty = one_sided_penalty
		self.distance_weighting = distance_weighting
		self.eps_drift = eps_drift
		self.eps_center = eps_center
		self.lambd_consistency_term = lambd_consistency_term


	def pre_train(self):
		if not self.did_init_train:
			self.train_init()

		self.zero_grad()
		self.optimizer.zero_grad()
		for p in self.parameters():
			p.requires_grad = True

	def update_parameters(self):
		self.optimizer.step()


	def train_batch(self, batch_real, batch_fake):
		"""
		Train discriminator for one batch of real and fake data

		Parameters
		----------
		batch_real : autograd.Variable
			Batch of real data
		batch_fake : autograd.Variable
			Batch of fake data

		Returns
		-------
		loss_real : float
			WGAN loss for real data
		loss_fake : float
			WGAN loss for fake data
		loss_penalty : float
			Improved WGAN penalty term
		loss_drift : float
			Drifting penalty
		loss_center : float
			Center penalty
		"""
		self.pre_train()

		one = torch.FloatTensor([1])
		mone = one * -1

		batch_real,one,mone = utils.cuda_check([batch_real,one,mone])

		fx_real = self(batch_real)
		loss_real = fx_real.mean().reshape(-1)
		loss_real.backward(mone,
						   retain_graph=(self.eps_drift>0 or self.eps_center>0))

		fx_fake = self(batch_fake)
		loss_fake = fx_fake.mean().reshape(-1)
		loss_fake.backward(one,
						   retain_graph=(self.eps_drift>0 or self.eps_center>0))

		loss_drift = 0
		loss_center = 0
		if self.eps_drift>0:
			tmp_drift = self.eps_drift*loss_real**2
			tmp_drift.backward(retain_graph=self.eps_center>0)
			loss_drift = tmp_drift.data[0]
		if self.eps_center>0:
			tmp_center = (loss_real+loss_fake)
			tmp_center = self.eps_center*tmp_center**2
			tmp_center.backward()
			loss_center = tmp_center.data[0]

		#loss_consistency_term
		#if self.lambd_consistency_term>0:
		#	batch_real_1

		dist = 1
		if self.distance_weighting:
			dist = (loss_real-loss_fake).detach()
			dist = dist.clamp(min=0)
		loss_penalty = self.calc_gradient_penalty(batch_real, batch_fake)
		loss_penalty = self.lambd*dist*loss_penalty
		loss_penalty.backward()

		# Update parameters
		self.update_parameters()

		loss_real = -loss_real.data[0]
		loss_fake = loss_fake.data[0]
		loss_penalty = loss_penalty.data[0]
		return loss_real,loss_fake,loss_penalty,loss_drift,loss_center # return loss


	def calc_gradient_penalty(self, batch_real, batch_fake):
		"""
		Improved WGAN gradient penalty

		Parameters
		----------
		batch_real : autograd.Variable
			Batch of real data
		batch_fake : autograd.Variable
			Batch of fake data

		Returns
		-------
		gradient_penalty : autograd.Variable
			Gradient penalties
		"""
		alpha = torch.rand(batch_real.data.size(0),*((len(batch_real.data.size())-1)*[1]))
		alpha = alpha.expand(batch_real.data.size())
		batch_real,alpha = utils.cuda_check([batch_real,alpha])

		interpolates = alpha * batch_real.data + ((1 - alpha) * batch_fake.data)
		interpolates = Variable(interpolates, requires_grad=True)
		alpha,interpolates = utils.cuda_check([alpha,interpolates])

		disc_interpolates = self(interpolates)

		ones = torch.ones(disc_interpolates.size())
		interpolates,ones = utils.cuda_check([interpolates,ones])

		gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
								  grad_outputs=ones,
								  create_graph=True, retain_graph=True, only_inputs=True)[0]
		gradients = gradients.view(gradients.size(0), -1)
		tmp = (gradients.norm(2, dim=1) - 1)
		if self.one_sided_penalty:
			tmp = tmp.clamp(min=0)
		gradient_penalty = ((tmp) ** 2).mean()

		return gradient_penalty


class WGAN_I_Generator(GAN_Module):
	"""
	Improved Wasserstein GAN generator

	References
	----------
	Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017).
	Improved Training of Wasserstein GANs.
	Retrieved from http://arxiv.org/abs/1704.00028
	"""
	def __init__(self):
		super(WGAN_I_Generator, self).__init__()

	def train_init(self,alpha=1e-4,betas=(0.5,0.9)):
		"""
		Initialize Adam optimizer for generator

		Parameters
		----------
		alpha : float, optional
			Learning rate for Adam
		betas : (float,float), optional
			Betas for Adam
		"""
		self.loss = None
		self.optimizer = optim.Adam(self.parameters(),lr=alpha,betas=betas)
		self.did_init_train = True

	def pre_train(self,discriminator):
		if not self.did_init_train:
			self.train_init()

		self.zero_grad()
		self.optimizer.zero_grad()
		for p in discriminator.parameters():
			p.requires_grad = False  # to avoid computation

	def update_parameters(self):
		self.optimizer.step()

	def train_batch(self, batch_noise, discriminator):
		"""
		Train generator for one batch of latent noise

		Parameters
		----------
		batch_noise : autograd.Variable
			Batch of latent noise
		discriminator : nn.Module
			Discriminator to evaluate realness of generated data

		Returns
		-------
		loss : float
			WGAN loss against evaluation of discriminator of generated samples
			to be real
		"""

		self.pre_train(discriminator)

		mone = torch.FloatTensor([1]) * -1
		batch_noise,mone = utils.cuda_check([batch_noise,mone])

		# Generate and discriminate
		gen = self(batch_noise)
		disc = discriminator(gen)
		loss = disc.mean().reshape(-1)
		# Backprop gradient
		loss.backward(mone)

		# Update parameters
		self.update_parameters()

		loss = loss.data[0]
		return loss # return loss
