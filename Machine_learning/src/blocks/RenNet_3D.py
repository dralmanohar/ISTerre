import torch


class Block(torch.nn.Module):
	
	def __init__(self,  in_channels, out_channels, identity_downsample = None, stride = 1):
		super(Block, self).__init__()
		
		self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
		self.bn1   = torch.nn.BatchNorm2d(out_channels)
		self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
		self.bn2   = torch.nn.BatchNorm2d(out_channels)
		self.relu  = torch.nn.ReLU()
		self.identity_downsample = identity_downsample
		
	def forward(self, x):
		identity = x
		
		x = self.conv1(x)
		x = self.relu(x)
		x = self.conv2(x)
		
		print ("x shape before = ", x.shape)
		
		if self.identity_downsample is not None:
			identity = self.identity_downsample(identity)
		
		print ("identity shape after = ", identity.shape)
		
		x +=identity
		
		x = self.relu(x)
		
		return x

class Resnet(torch.nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Resnet, self).__init__()
		
		self.init_channels = 4
		
		self.conv1   = torch.nn.Conv2d(in_channels, 64, kernel_size = 7, stride = 2, padding = 3)
		self.bn1     = torch.nn.BatchNorm2d(64)
		self.relu    = torch.nn.ReLU()
		self.maxpool = torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1) 
		
		
		### resnet layers 
		
		self.layer1 = self.__make_layer(64, 64, stride = 1)
		
		self.layer2 = self.__make_layer(64, 128, stride = 2)
		
		# ~ self.layer3 = self.__make_layer(128, 256, stride = 2)
		
		# ~ self.layer4 = self.__make_layer(256, 512, stride = 2)
		
		self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
		
		self.fc      = torch.nn.Linear(512, out_channels)
		
	def __make_layer(self, in_dim, out_dim, stride):
		
		identity_downsample = None
		
		if stride !=1:
			
			identity_downsample = self.identity_downsample(in_dim, out_dim)
			
		return torch.nn.Sequential(
				
				Block(in_dim, out_dim, identity_downsample = identity_downsample, stride = stride),
				
				Block(out_dim, out_dim)
				
				)
	
	def forward(self, x):
		
		x = self.conv1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		
		x = self.layer1(x)
		x = self.layer2(x)
		# ~ x = self.layer3(x)
		# ~ x = self.layer4(x)
		
		# ~ x = self.avgpool(x)
		
		# ~ x = x.view(x.shape[0],-1)
		
		# ~ x = self.fc(x)
		
		return x
		
	
	
		
	def identity_downsample(self, in_dim, out_dim):
		return torch.nn.Sequential(
			
			torch.nn.Conv2d(in_dim, out_dim, kernel_size = 3, stride = 2, padding = 1)
		)
		 

