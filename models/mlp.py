#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.7
import torch
from torch import nn
import torch.nn.functional as F

IMAGE_SIZE = 28
IMAGE_CNANNEL = 1

class Model(nn.Module):
	''' 
		MLP model for MNIST dataset.
	'''

	def __init__(self):
		super(Model, self).__init__()
		dim_in	= IMAGE_CNANNEL*IMAGE_SIZE*IMAGE_SIZE
		dim_out = 10

		self.layer_hidden1 = nn.Linear(in_features=dim_in, out_features=200)
		self.layer_hidden2 = nn.Linear(in_features=200, out_features=200)
		self.layer_out = nn.Linear(in_features=200, out_features=dim_out)

	def forward(self, x):
		x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
		
		## two fully connected layers
		x = F.relu(self.layer_hidden1(x))
		x = F.relu(self.layer_hidden2(x))

		## final softmax output layer
		x = F.log_softmax(self.layer_out(x), dim=1)