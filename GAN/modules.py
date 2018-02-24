import torch
from torch.nn.modules.module import Module
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F

class Upsample(Module):
	def __init__(self,channels,scale_factor):
		self.scale_factor = scale_factor
		self.channels = channels
		super(Upsample, self).__init__()

		self.register_buffer('_ones', torch.Tensor(self.channels,1,*self.scale_factor))
		self._ones.data.fill_(1)

	def forward(self,input):
		if input.is_cuda:
			self._ones = self._ones.cuda()
		else:
			self._ones = self._ones.cpu()
		return F.conv_transpose2d(
			input, self._ones, stride=self.scale_factor, groups=self.channels)


class LayerNorm(Module):
	def __init__(self,num_features,n_dim,eps=1e-5,affine=True):
		assert(n_dim>1)

		super(LayerNorm, self).__init__()
		self.num_features = num_features
		self.n_dim = n_dim

		tmp_ones = [1]*(n_dim-2)
		self.affine = affine
		self.eps = eps
		if self.affine:
			self.weight = Parameter(torch.Tensor(1,num_features,*tmp_ones))
			self.weight.data.fill_(1)
			self.bias = Parameter(torch.Tensor(1,num_features,*tmp_ones))
			self.weight.data.fill_(0)
		else:
			self.register_parameter('weight', None)
			self.register_parameter('bias', None)

		self.reset_parameters()

	def reset_parameters(self):
		if self.affine:
			self.weight.data.uniform_()
			self.bias.data.zero_()

	def forward(self, input):
		orig_size = input.size()
		b = orig_size[0]
		tmp_dims = range(self.n_dim)

		trash_mean = torch.zeros(b)
		trash_var = torch.ones(b)
		if input.is_cuda:
			trash_mean = trash_mean.cuda()
			trash_var = trash_var.cuda()

		input_reshaped = input.contiguous().permute(1,0,*tmp_dims[2:]).contiguous()

		out = F.batch_norm(
			input_reshaped, trash_mean, trash_var, None, None,
			True, 0., self.eps).permute(1,0,*tmp_dims[2:]).contiguous()

		if self.affine:
			weight = self.weight
			bias = self.bias
			out = weight*out+bias

		return out


class Conv2d_contiguous(nn.Conv2d):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding=0, dilation=1, groups=1, bias=True):
		super(Conv2d_contiguous, self).__init__(
			in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

	def forward(self, input):
		out = super(Conv2d_contiguous, self).forward(input)
		return out.contiguous()


class PixelShuffle1d(Module):
	def __init__(self,scale_kernel):
		super(PixelShuffle1d, self).__init__()
		self.scale_kernel = scale_kernel

	def forward(self, input):
		batch_size, channels, in_height = input.size()
		channels //= self.scale_kernel[0]

		out_height = in_height * self.scale_kernel[0]

		input_view = input.contiguous().view(
			batch_size, channels, self.scale_kernel[0],in_height)

		shuffle_out = input_view.permute(0, 1, 3, 2).contiguous()
		return shuffle_out.view(batch_size, channels, out_height)

class PixelShuffle2d(Module):
	def __init__(self,scale_kernel):
		super(PixelShuffle2d, self).__init__()
		self.scale_kernel = scale_kernel

	def forward(self, input):
		batch_size, channels, in_height, in_width = input.size()
		channels //= self.scale_kernel[0]*self.scale_kernel[1]

		out_height = in_height * self.scale_kernel[0]
		out_width = in_width * self.scale_kernel[1]

		input_view = input.contiguous().view(
			batch_size, channels, self.scale_kernel[0], self.scale_kernel[1],
			in_height, in_width)

		shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
		return shuffle_out.view(batch_size, channels, out_height, out_width)


class RobinShuffle2d(Module):
	"""
	IIIIIIIIII

	xxxxx
	 ooooo
	  +++++
	   mmmmm
		nnnnn
		 xxxxx

	  xo+mnx
	"""

	def __init__(self, in_channels, out_channels, kernel_size, bias=True):
		super(RobinShuffle2d, self).__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size

		self.weight = Parameter(torch.Tensor(
				np.prod(out_channels*np.asarray(kernel_size)), in_channels, *kernel_size))

		if bias:
			self.bias = Parameter(torch.Tensor(np.sum(out_channels*kernel_size)))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		n = self.in_channels
		for k in self.kernel_size:
			n *= k
		stdv = 1. / np.sqrt(n)
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input):
		out_size = np.asarray(input.size())
		out_size[1] = self.out_channels
		out_size[2] = out_size[2]-(self.kernel_size[0]-1)
		out_size[3] = out_size[3]-(self.kernel_size[1]-1)

		output = Variable(torch.Tensor(*out_size),requires_grad=input.requires_grad,volatile=input.volatile)
		if input.is_cuda:
			output = output.cuda()
		for i in range(self.kernel_size[0]):
			for j in range(self.kernel_size[1]):
				for z in range(self.out_channels):
					out_indeces_1 = torch.from_numpy(np.arange(i,input.size(2)-self.kernel_size[0]+1,self.kernel_size[0]))
					out_indeces_2 = torch.from_numpy(np.arange(j,input.size(3)-self.kernel_size[1]+1,self.kernel_size[1]))

					bias = None
					if self.bias is not None:
						bias = self.bias[(i*self.kernel_size[1]+j)+z*(self.kernel_size[0]*self.kernel_size[1])]

					tmp = F.conv2d(input[:,:,i:,j:],
							self.weight[[(i*self.kernel_size[1]+j)+z*(self.kernel_size[0]*self.kernel_size[1])],:],
							bias,
							self.kernel_size)

					for x_i,x in enumerate(out_indeces_1):
						for y_i,y in enumerate(out_indeces_2):
							output[:,z,x,y] = tmp[:,0,x_i,y_i]

		return output


class PrintSize(Module):
	def __init__(self):
		super(PrintSize, self).__init__()

	def forward(self, input):
		print(input.size())
		return input


class Dummy(Module):
	def __init__(self):
		super(Dummy, self).__init__()

	def forward(self, input):
		return input


class ACGAN_Latent(nn.Linear):
	def __init__(self,n_input,n_output,n_classes):
		super(ACGAN_Latent, self).__init__(n_input+n_classes,n_output)

	def forward(self,input,targets):
		input = torch.cat((input,targets),1)
		output = super(ACGAN_Dense, self).forward(input)
		return output


class ACGAN_Dense(nn.Linear):
	def __init__(self,n_input,n_classes):
		super(ACGAN_Dense, self).__init__(n_input,1+n_classes)

	def forward(self, input):
		output = super(ACGAN_Dense, self).forward(input)
		return output[:,0].contiguous(),output[:,1:].contiguous()
