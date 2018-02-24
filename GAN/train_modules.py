import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.autograd as autograd
from gpustat import print_gpustat
import torch.nn.functional as F

class GAN_Module(nn.Module):
	""" Standard module for different GAN implementations """
	def __init__(self):
		super(GAN_Module, self).__init__()

		self.did_init_train = False

	def save_model(self,fname):
		model_state = self.state_dict()
		opt_state = self.optimizer.state_dict()

		torch.save((model_state,opt_state,self.did_init_train),fname)

	def load_model(self,fname):
		model_state,opt_state,self.did_init_train = torch.load(fname)

		self.load_state_dict(model_state)
		self.optimizer.load_state_dict(opt_state)


class GAN_Discriminator(GAN_Module):
	""" Standard GAN discriminator """
	def __init__(self):
		super(GAN_Discriminator, self).__init__()

		self.did_init_train = False

	def train_init(self,alpha=1e-4,betas=(0.5,0.9),soft_labels=False):
		self.optimizer = optim.Adam(self.parameters(),lr=alpha,betas=betas)
		self.loss = torch.nn.BCELoss()
		self.did_init_train = True
		self.soft_labels = soft_labels

	def train_batch(self, batch_real, batch_fake, cuda=True):
		if not self.did_init_train:
			self.train_init()

		self.zero_grad()
		self.optimizer.zero_grad()
		for p in self.parameters():
			p.requires_grad = True

		# Data has to be torch variable
		real = batch_real
		fake = batch_fake


		ones_label = torch.ones(real.size(0),1)
		zeros_label = torch.zeros(fake.size(0),1)

		if self.soft_labels:
			ones_label += torch.randn(ones_label.size())*0.1
			zeros_label += torch.randn(zeros_label.size()).abs()*0.1

		ones_label = Variable(ones_label)
		zeros_label = Variable(zeros_label)

		if cuda:
			ones_label = ones_label.cuda()
			zeros_label = zeros_label.cuda()


		# Compute output and loss
		fx_real = self.forward(real)
		loss_real = self.loss.forward(fx_real,ones_label)
		loss_real.backward()
		fx_fake = self.forward(fake)
		loss_fake = self.loss.forward(fx_fake,zeros_label)
		loss_fake.backward()

		#loss = loss_real + loss_fake

		# Update parameters
		self.optimizer.step()

		return (loss_fake.data[0],loss_real.data[0],0) # return loss


class GAN_Generator(GAN_Module):
	""" Standard GAN generator """
	def __init__(self):
		super(GAN_Generator, self).__init__()

		self.did_init_train = False

	def train_init(self,alpha=1e-4,betas=(0.5,0.9),soft_labels=False):
		self.optimizer = optim.Adam(self.parameters(),lr=alpha,betas=betas)
		self.loss = torch.nn.BCELoss()
		self.did_init_train = True
		self.soft_labels = soft_labels

	def train_batch(self, batch_noise, discriminator, cuda=True):
		if not self.did_init_train:
			self.train_init()

		self.zero_grad()
		self.optimizer.zero_grad()
		for p in discriminator.parameters():
			p.requires_grad = False  # to avoid computation

		noise = batch_noise

		# Generate and discriminate
		gen = self.forward(noise)
		disc = discriminator(gen)


		ones_label = torch.ones(disc.size())

		if self.soft_labels:
			ones_label += torch.randn(ones_label.size())*0.1

		ones_label = Variable(ones_label)

		if cuda:
			ones_label = ones_label.cuda()

		loss = self.loss.forward(disc,ones_label)

		# Backprop gradient
		loss.backward()

		# Update parameters
		self.optimizer.step()

		return loss.data[0] # return loss


class GAN_Discriminator_SoftPlus(GAN_Module):
	""" Standard GAN discriminator """
	def __init__(self):
		super(GAN_Discriminator_SoftPlus, self).__init__()

		self.did_init_train = False

	def train_init(self,alpha=1e-4,betas=(0.5,0.9),soft_labels=False):
		self.optimizer = optim.Adam(self.parameters(),lr=alpha,betas=betas)
		self.loss = None
		self.did_init_train = True
		self.soft_labels = soft_labels

	def train_batch(self, batch_real, batch_fake, cuda=True):
		if not self.did_init_train:
			self.train_init()

		self.zero_grad()
		self.optimizer.zero_grad()
		for p in self.parameters():
			p.requires_grad = True

		# Data has to be torch variable
		real = batch_real
		fake = batch_fake

		# Compute output and loss
		fx_real = self.forward(real)
		loss_real = F.softplus(-fx_real).mean()
		loss_real.backward()

		fx_fake = self.forward(fake)
		loss_fake = F.softplus(fx_fake).mean()
		loss_fake.backward()

		# Update parameters
		self.optimizer.step()

		return (loss_fake.data[0],loss_real.data[0],0) # return loss


class GAN_Generator_SoftPlus(GAN_Module):
	""" Standard GAN generator """
	def __init__(self):
		super(GAN_Generator_SoftPlus, self).__init__()

		self.did_init_train = False

	def train_init(self,alpha=1e-4,betas=(0.5,0.9),soft_labels=False):
		self.optimizer = optim.Adam(self.parameters(),lr=alpha,betas=betas)
		self.loss = None
		self.did_init_train = True
		self.soft_labels = soft_labels

	def train_batch(self, batch_noise, discriminator, cuda=True):
		if not self.did_init_train:
			self.train_init()

		self.zero_grad()
		self.optimizer.zero_grad()
		for p in discriminator.parameters():
			p.requires_grad = False  # to avoid computation

		noise = batch_noise

		# Generate and discriminate
		gen = self.forward(noise)
		disc = discriminator(gen)

		loss = F.softplus(-disc).mean()

		# Backprop gradient
		loss.backward()

		# Update parameters
		self.optimizer.step()

		return loss.data[0] # return loss


class ACGAN_Discriminator_SoftPlus(GAN_Module):
	""" Standard GAN discriminator """
	def __init__(self):
		super(ACGAN_Discriminator_SoftPlus, self).__init__()

		self.did_init_train = False

	def train_init(self,alpha=1e-4,betas=(0.5,0.9),soft_labels=False):
		self.optimizer = optim.Adam(self.parameters(),lr=alpha,betas=betas)
		self.loss = nn.CrossEntropyLoss()
		self.did_init_train = True
		self.soft_labels = soft_labels

	def train_batch(self, batch_real, batch_fake, c_real, c_fake, cuda=True):
		if not self.did_init_train:
			self.train_init()

		self.zero_grad()
		self.optimizer.zero_grad()
		for p in self.parameters():
			p.requires_grad = True

		# Data has to be torch variable
		real = batch_real
		fake = batch_fake

		# Compute output and loss
		fx_real, fx_c_real = self(batch_real)
		loss_real = F.softplus(-fx_real).mean()
		loss_real.backward(retain_graph=True)

		fx_fake, fx_c_fake = self(batch_fake)
		loss_fake = F.softplus(fx_fake).mean()
		loss_fake.backward(retain_graph=True)

		loss_class = self.loss.forward(fx_c_real,c_real)
		loss_class += self.loss.forward(fx_c_fake,c_fake)
		loss_class /= 2.
		loss_class.backward()

		# Update parameters
		self.optimizer.step()

		return (loss_fake.data[0],loss_real.data[0],loss_class.data[0]) # return loss


class ACGAN_Generator_SoftPlus(GAN_Module):
	""" Standard GAN generator """
	def __init__(self):
		super(ACGAN_Generator_SoftPlus, self).__init__()

		self.did_init_train = False

	def train_init(self,alpha=1e-4,betas=(0.5,0.9),soft_labels=False):
		self.optimizer = optim.Adam(self.parameters(),lr=alpha,betas=betas)
		self.loss = nn.CrossEntropyLoss()
		self.did_init_train = True
		self.soft_labels = soft_labels

	def train_batch(self, batch_noise, batch_c, batch_c_onehot, discriminator, cuda=True):
		if not self.did_init_train:
			self.train_init()

		self.zero_grad()
		self.optimizer.zero_grad()
		for p in discriminator.parameters():
			p.requires_grad = False  # to avoid computation

		noise = batch_noise

		# Generate and discriminate
		gen = self.forward(noise,batch_c_onehot)
		fx,fx_c = discriminator(gen)

		loss = F.softplus(-fx).mean()
		loss.backward(retain_graph=True)

		loss_class = self.loss.forward(fx_c,batch_c)
		loss_class.backward()

		# Update parameters
		self.optimizer.step()

		return loss.data[0],loss_class.data[0] # return loss



class WGAN_Discriminator(GAN_Module):
	""" Standard WGAN discriminator """
	def __init__(self):
		super(WGAN_Discriminator, self).__init__()

		self.did_init_train = False

	def train_init(self, lr=0.0001, c=0.01):
		self.c = c
		for p in self.parameters():
			p.data.clamp_(-self.c,self.c)

		self.loss = None
		self.optimizer = optim.RMSprop(self.parameters(),lr=lr)
		self.did_init_train = True

	def train_batch(self, batch_real, batch_fake):
		if not self.did_init_train:
			self.train_init()

		# Data has to be torch variable
		real = batch_real
		fake = batch_fake

		# Reset gradients
		self.optimizer.zero_grad()

		# Compute output and loss
		fx_real = self.forward(real)
		loss_real = -torch.mean(fx_real)
		loss_real.backward()

		fx_fake = self.forward(fake)
		loss_fake = torch.mean(fx_fake)
		loss_fake.backward()

		loss = loss_real + loss_fake

		# Backprop gradient
		#loss.backward()

		# Update parameters
		self.optimizer.step()

		for p in self.parameters():
			p.data.clamp_(-self.c,self.c)

		# Better safe than sorry
		self.optimizer.zero_grad()

		return loss.data[0] # return loss


class WGAN_Generator(GAN_Module):
	""" Standard WGAN generator """
	def __init__(self):
		super(WGAN_Generator, self).__init__()

		self.did_init_train = False

	def train_init(self, lr=0.0001):
		self.loss = None
		self.optimizer = optim.RMSprop(self.parameters(),lr=lr)
		self.did_init_train = True

	def train_batch(self, batch_noise, discriminator):
		if not self.did_init_train:
			self.train_init()

		# Reset gradients
		#self.optimizer.zero_grad()

		# Generate and discriminate
		gen = self.forward(batch_noise)
		disc = discriminator(gen)
		loss = -torch.mean(disc)

		# Backprop gradient
		loss.backward()

		# Update parameters
		self.optimizer.step()

		# Better safe than sorry
		self.optimizer.zero_grad()
		discriminator.zero_grad()

		return loss.data[0] # return loss



class WGAN_I_Discriminator(GAN_Module):
	def __init__(self):
		super(WGAN_I_Discriminator, self).__init__()

		self.did_init_train = False

	def train_init(self,alpha=1e-4,betas=(0.5,0.9)):
		self.loss = None
		self.optimizer = optim.Adam(self.parameters(),lr=alpha,betas=betas)
		self.did_init_train = True

	def train_batch(self, batch_real, batch_fake, lambd=10, cuda=True):
		if not self.did_init_train:
			self.train_init()

		self.zero_grad()
		self.optimizer.zero_grad()
		for p in self.parameters():
			p.requires_grad = True

		one = torch.FloatTensor([1])
		if cuda:
			one = one.cuda()
		mone = one * -1

		#D_real = netD(real_data_v)
		#D_real = D_real.mean()
		#D_real.backward(mone)

		fx_real = self(batch_real)
		loss_real = fx_real.mean()
		loss_real.backward(mone)
		loss_r = loss_real.data[0]


		#D_fake = netD(inputv)
		#D_fake = D_fake.mean()
		#D_fake.backward(one)

		fx_fake = self(batch_fake)
		loss_fake = fx_fake.mean()
		loss_fake.backward(one)
		loss_f = loss_fake.data[0]

		#dreist geklaut von
		# https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
		# gradients = autograd.grad(outputs=fx_comb, inputs=interpolates,
		# 					  grad_outputs=grad_ones,
		# 					  create_graph=True, retain_graph=True, only_inputs=True)[0]
		# gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambd
		loss_penalty = self.calc_gradient_penalty(batch_real, batch_fake,lambd,cuda)
		loss_penalty.backward()

		#loss = loss_fake + loss_real + loss_penalty

		# Backprop gradient
		penalty = loss_penalty.data[0]
		#del loss_real,loss_fake,loss_penalty
		#loss.backward()

		# Update parameters
		self.optimizer.step()

		return (loss_f,loss_r,penalty) # return loss


	def calc_gradient_penalty(self, real_data, fake_data,lambd,cuda=True):
		alpha = torch.rand(real_data.data.size(0),*((len(real_data.data.size())-1)*[1]))
		alpha = alpha.expand(real_data.data.size())
		if cuda:
			alpha = alpha.cuda()

		interpolates = alpha * real_data.data + ((1 - alpha) * fake_data.data)

		if cuda:
			interpolates = interpolates.cuda()
		interpolates = Variable(interpolates, requires_grad=True)

		disc_interpolates = self(interpolates)

		ones = torch.ones(disc_interpolates.size())
		if cuda:
			ones = ones.cuda()

		gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
								  grad_outputs=ones,
								  create_graph=True, retain_graph=True, only_inputs=True)[0]

		gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambd
		return gradient_penalty


class WGAN_I_Generator(GAN_Module):
	def __init__(self):
		super(WGAN_I_Generator, self).__init__()

		self.did_init_train = False

	def train_init(self,alpha=1e-4,betas=(0.5,0.9)):
		self.loss = None
		self.optimizer = optim.Adam(self.parameters(),lr=alpha,betas=betas)
		self.did_init_train = True

	def train_batch(self, batch_noise, discriminator, cuda=True):
		if not self.did_init_train:
			self.train_init()

		self.zero_grad()
		self.optimizer.zero_grad()
		for p in discriminator.parameters():
			p.requires_grad = False  # to avoid computation

		noise = batch_noise
		mone = torch.FloatTensor([1]) * -1
		if cuda:
			mone = mone.cuda()

		#one = torch.FloatTensor([1]).cuda()
		#mone = one * -1

		#fake = netG(noisev, real_data_v)
		#G = netD(fake)
		#G = G.mean()
		#G.backward(mone)

		# Generate and discriminate
		gen = self(noise)
		disc = discriminator(gen)
		loss = disc.mean()
		# Backprop gradient
		loss.backward(mone)

		# Update parameters
		self.optimizer.step()

		return loss.data[0] # return loss


class ACWGAN_I_Discriminator(WGAN_I_Discriminator):

	def train_init(self,alpha=1e-4,betas=(0.5,0.9),weight_decay=5e-5):
		self.loss = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(self.parameters(),lr=alpha,betas=betas,weight_decay=weight_decay)
		self.did_init_train = True


	def calc_gradient_penalty(self, real_data, fake_data,lambd):
		alpha = torch.rand(real_data.data.size(0),*((len(real_data.data.size())-1)*[1]))
		alpha = alpha.expand(real_data.data.size())
		alpha = alpha.cuda()

		interpolates = alpha * real_data.data + ((1 - alpha) * fake_data.data)

		interpolates = interpolates.cuda()
		interpolates = Variable(interpolates, requires_grad=True)

		disc_interpolates,_ = self(interpolates)

		gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
								  grad_outputs=torch.ones(disc_interpolates.size()).cuda() ,
								  create_graph=True, retain_graph=True)[0]

		gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambd
		return gradient_penalty

	def train_batch(self, batch_real, batch_fake, c_real, c_fake, lambd=10, cuda=True):
		if not self.did_init_train:
			self.train_init()

		self.zero_grad()
		self.optimizer.zero_grad()
		for p in self.parameters():
			p.requires_grad = True

		one = torch.FloatTensor([1]).cuda()
		mone = one * -1

		fx_real, fx_c_real = self(batch_real)
		loss_real = fx_real.mean()
		loss_real.backward(mone,retain_graph=True)
		loss_r = loss_real.data[0]

		fx_fake, fx_c_fake = self(batch_fake)
		loss_fake = fx_fake.mean()
		loss_fake.backward(one,retain_graph=True)
		loss_f = loss_fake.data[0]

		loss_class = self.loss.forward(fx_c_real,c_real)
		loss_class += self.loss.forward(fx_c_fake,c_fake)
		loss_class /= 2.
		loss_class.backward(one)
		loss_c = loss_class.data[0]

		#dreist geklaut von
		# https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
		# gradients = autograd.grad(outputs=fx_comb, inputs=interpolates,
		# 					  grad_outputs=grad_ones,
		# 					  create_graph=True, retain_graph=True, only_inputs=True)[0]
		# gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambd
		loss_penalty = self.calc_gradient_penalty(batch_real, batch_fake,lambd)
		loss_penalty.backward()
		penalty = loss_penalty.data[0]
		#penalty = 0
		# Update parameters
		self.optimizer.step()

		#self.zero_grad()
		#self.optimizer.zero_grad()
		del loss_penalty
		return (loss_f,loss_r,penalty,loss_c) # return loss

class ACWGAN_I_Generator(WGAN_I_Generator):

	def train_init(self,alpha=1e-4,betas=(0.5,0.9),weight_decay=5e-5):
		self.loss = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(self.parameters(),lr=alpha,betas=betas,weight_decay=weight_decay)
		self.did_init_train = True

	def train_batch(self, batch_noise, batch_c, batch_c_onehot, discriminator, cuda=True):
		if not self.did_init_train:
			self.train_init()

		self.zero_grad()
		self.optimizer.zero_grad()
		for p in discriminator.parameters():
			p.requires_grad = False  # to avoid computation

		noise = batch_noise
		one = torch.FloatTensor([1]).cuda()
		mone = one * -1

		# Generate and discriminate
		gen = self(noise,batch_c_onehot)
		fx,fx_c = discriminator(gen)
		loss = fx.mean()
		# Backprop gradient
		loss.backward(mone,retain_graph=True)

		loss_class = self.loss.forward(fx_c,batch_c)
		loss_class.backward(one)
		loss_c = loss_class.data[0]

		# Update parameters
		self.optimizer.step()

		#self.zero_grad()
		#self.optimizer.zero_grad()
		#discriminator.zero_grad()
		#discriminator.optimizer.zero_grad()

		return loss.data[0],loss_c # return loss
