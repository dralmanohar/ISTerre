import torch
import numpy as np
import torch.nn.functional as F


class Conv_FC(torch.nn.Module):
	def __init__(self, in_size, in_dim, out_dim, kernel=3, prob=0):
		super(Conv_FC, self).__init__()
		
		self.x1 = in_size
		self.k  = kernel
		self.p  = prob
		self.input_dim  = in_dim
		self.output_dim = out_dim
		
		self.conv1 = torch.nn.Conv3d(in_size, 64, kernel, stride=1, padding= 1, bias=False)
		torch.nn.init.kaiming_normal_(self.conv1.weight)
		self.maxpool1 = torch.nn.MaxPool3d((2,2,2))
		
		self.conv2 = torch.nn.Conv3d(64, 128, kernel, stride=1, padding= 1, bias=False)
		torch.nn.init.kaiming_normal_(self.conv2.weight)
		self.maxpool2 = torch.nn.MaxPool3d((2,2,2))
		
		
		self.conv3 = torch.nn.Conv3d(128, 512, kernel, stride=1, padding= 1, bias=False)
		torch.nn.init.kaiming_normal_(self.conv3.weight)
		self.maxpool3 = torch.nn.MaxPool3d((2,2,2))
		
		self.fc_1 = torch.nn.Linear(out_dim, out_dim)
		self.fc_2 = torch.nn.Linear(out_dim, out_dim)
		
		# ~ self.dropout = torch.nn.Dropout(prob)
		
	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.conv3(x)
		x = F.relu(x)
		
		# ~ print ("shape before =\t",x.shape)
		# ~ print ("in_dim =\t",in_dim)
		
		x = x.view(x.shape[0],-1)
		
		# ~ print ("shape after =\t",x.shape)
		
		x = self.fc_1(x)
		x = F.relu(x)
		x = self.fc_2(x)
		
		return x
		
		# ~ x = self.dropout(x)
		# ~ return x #torch.nn.functional.relu(x)  #, inplace=True)
