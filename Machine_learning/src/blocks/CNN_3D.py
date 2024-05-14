import torch
import numpy as np

class ConvUnit(torch.nn.Module):
	def __init__(self, in_size, out_size, kernel=3, prob=0):
		super(ConvUnit, self).__init__()
		
		self.x = in_size
		self.y = out_size
		self.k = kernel
		self.p = prob
		
		self.conv = torch.nn.Conv3d(in_size, out_size, kernel, stride=1, padding= 1, bias=False)
		self.relu = torch.nn.ReLU()
		torch.nn.init.kaiming_normal_(self.conv.weight)
		
	def forward(self, x):
		x = self.conv(x)
		x = self.relu(x)
		return x 

class UnitaryUnit(torch.nn.Module):
	
	def __init__(self, in_size):
		super(UnitaryUnit, self).__init__()
		
		self.x = in_size
		self.k = 1
		self.conv = torch.nn.Conv3d(in_size, 1, 1, stride=1, padding=0, bias=False)
		
	def forward(self, x):
		x = self.conv(x)
		return x
		
class BlockCNN(torch.nn.Module):
	def __init__(self, name, layers):
		super(BlockCNN, self).__init__()
		self.name = name
		self.layers = torch.nn.ModuleList(layers)
	
	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x
