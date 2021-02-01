#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.7
import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
	''' 
		The CNN model for CIFAR10 dataset.
		Build according to http://www.tensorfly.cn/tfdoc/tutorials/deep_cnn.html
	'''
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
							   kernel_size=5, padding=2)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
		self.bn1 = nn.BatchNorm2d(32,eps=1e-5,momentum=.1,affine=True,track_running_stats=True)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
		self.bn2 = nn.BatchNorm2d(64,eps=1e-5,momentum=.1,affine=True,track_running_stats=True)
		self.bn3 = nn.BatchNorm2d(128,eps=1e-5,momentum=.1,affine=True,track_running_stats=True)
		self.drop = nn.Dropout(p=0.5)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
		self.dense= nn.Linear(in_features=2048, out_features=10)


	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.pool1(x)
		x = self.bn1(x)
		x = self.drop(x)

		x = F.relu(self.conv2(x))
		x = self.pool2(x)
		x = self.bn2(x)
		x = self.drop(x)
        
		x = F.relu(self.conv3(x))
		x = self.pool2(x)
		x = self.bn3(x)
		x = self.drop(x)
        
		x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
		x = F.log_softmax(self.dense(x), dim=1)

		return x

# 	def __init__(self):
# 		super(Model, self).__init__()
# 		self.conv1 = nn.Conv2d(3, 6, 5)
# 		self.pool = nn.MaxPool2d(2, 2)
# 		self.bn1 = nn.BatchNorm2d(6,eps=1e-5,momentum=.1,affine=True,track_running_stats=True)
# 		self.conv2 = nn.Conv2d(6, 16, 5)
# 		self.bn2 = nn.BatchNorm2d(16, eps=1e-5, momentum=.1, affine=True, track_running_stats=True)
# 		self.fc1 = nn.Linear(16 * 5 * 5, 120)
# 		self.fc2 = nn.Linear(120, 84)
# 		self.fc3 = nn.Linear(84, 10)
# 		for m in self.modules():
# 			if isinstance(m, (nn.Conv2d,nn.Linear)):
# 				nn.init.xavier_uniform_(m.weight)
	
# 	def forward(self, x):
# 		x = self.bn1(self.pool(F.relu(self.conv1(x))))
# 		x = self.bn2(self.pool(F.relu(self.conv2(x))))
# 		x = x.view(-1, 16 * 5 * 5)
# 		x = F.relu(self.fc1(x))
# 		x = F.relu(self.fc2(x))
# 		x = self.fc3(x)
# 		return F.log_softmax(x, dim=1)
	''' 
		The CNN model for CIFAR10 dataset.
		Build according to https://www.tensorflow.org/tutorials/images/cnn
	'''
# 	def __init__(self):
# 		super(Model, self).__init__()
# 		self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
# 							   kernel_size=3)
# 		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
# 		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
# 		self.bn1 = nn.BatchNorm2d(32,eps=1e-5,momentum=.1,affine=True,track_running_stats=True)
# 		self.bn2 = nn.BatchNorm2d(64,eps=1e-5,momentum=.1,affine=True,track_running_stats=True)
# 		self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding = 1)
# 		self.fc1 = nn.Linear(in_features=1024, out_features=64)
# 		self.fc2 = nn.Linear(in_features=64, out_features=10)


# 	def forward(self, x):
# 		x = F.relu(self.conv1(x))
# 		x = self.bn1(self.pool(x))
# 		x = F.relu(self.conv2(x))
# 		x = self.bn2(self.pool(x))
# 		x = F.relu(self.conv3(x))
# 		x = x.view(-1, 1024)
# 		x = F.relu(self.fc1(x))
# 		x = self.fc2(x)

# 		return F.log_softmax(x, dim=1)