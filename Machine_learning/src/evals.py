import torch

import numpy as np

import scipy
import scipy.stats
import scipy.spatial.distance

import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mse(x, y):
	res = (x - y)**2
	return res.mean()

def diss(divtau, theta):
	res = divtau * theta
	return res.mean()

def evaluate(dataset, models):
	tests_batch = 1
	
	
	# ~ print ("eval dataset =\t",dataset.inputs.shape)
	
	tests_loader = torch.utils.data.DataLoader(dataset, batch_size=tests_batch, shuffle=False)

	models.eval()
	
	with torch.no_grad():
		for step, batch in enumerate(tests_loader):
			data, labs = batch
			print ("step =\t",labs.shape)
			preds = models(data)
			mse_eval = mse(preds, labs)

	return preds, mse_eval
