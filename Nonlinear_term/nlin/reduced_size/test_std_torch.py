import numpy as np
import torch


x = torch.tensor([[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]],[[[20,14,15],[16,17,18]],[[19,40,21],[22,46,24]]]], dtype = torch.float32)

mean_torch = torch.mean(x, dim = [0, 2, 3])
std_torch  = torch.std(x,  dim = (0, 2, 3), unbiased=False)

print ("mean torch = \t", mean_torch)
print ("std torch = \t", std_torch)

mean_np = []
std_np = []

x1 = x.cpu().detach().numpy()

mean_f = np.zeros(x1.shape[0])

for i in range(x1.shape[1]):
	count = 0
	mean = np.zeros(x1.shape[0])
	for j in range(x1.shape[0]):
		mean[j] = np.mean(x1[j,i])
	
	mean_f[i] = np.mean(mean)

print ("mean final = \t",mean_f)

std_f = np.zeros(x1.shape[1])

for j in range(x1.shape[1]):
	std = 0
	count = 0
	for k in range(x1.shape[0]):
		for l in range(x1.shape[2]):
			for m in range(x1.shape[3]):
				std  += (x1[k,j, l, m] - mean_f[j])**2
				count +=1
	
	std_f[j] = np.sqrt(std/count)
print ("count = \t", std_f)
	
	
