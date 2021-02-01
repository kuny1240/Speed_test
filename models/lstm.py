#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.7
import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
	''' The stacked LSTM model for shakespear dataset. (2 layers).'''

	def __init__(self, dev):
		# 'char_num' is the number of all possible char in the dataset
		super(Model, self).__init__()
		char_num = 26

		self.lstmLayer		= 2
		self.embeddingDIM	= 8
		self.hiddenDIM		= 256
		self.device			= dev

		self.embedding = nn.Embedding(char_num, self.embeddingDIM)
		self.lstm = nn.LSTM(batch_first=True, input_size=self.embeddingDIM,
						hidden_size=self.hiddenDIM, num_layers=self.lstmLayer)
		self.linear=nn.Linear(in_features=self.hiddenDIM, out_features=char_num)

	def init_hidden(self, batchSize):
		h0 = torch.randn(self.lstmLayer, batchSize, self.hiddenDIM,
						 device = self.device)
		c0 = torch.randn(self.lstmLayer, batchSize, self.hiddenDIM,
						 device = self.device)
		return (h0, c0)

	def forward(self, seq):
		## embedding layer
		x = self.embedding(seq)
		
		## initial the hidden states firstly
		hiddenIn = self.init_hidden(x.shape[0])
		
		## stacked lstm layer
		x, hidden = self.lstm(x, hiddenIn)
		
		## output layer
		x = self.linear(x[:,-1,:])
		log_prob = F.log_softmax(x, dim=1)
		return log_prob