import sys

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import os
import torch


sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0]))

path = sys.path[0]

pathsrc = os.path.split(path)[0]#path.split[0]

srcpath = os.path.join(pathsrc, 'src')

sys.path.append(srcpath)

from dataset import Dataset, Standarization
from evals import evaluate

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

print ('Using device:',device)

# ~ ## import testing data

#data_path = '/bettik/PROJECTS/pr-geodyn_nn/COMMON/900_samples/'
#file_path = '/home/sharmam-ext/model/3D_xshells_data/32_32_256/movie3/900_samples/f_uu/r_comp/notebooks/model_unet/'

#print ("path = \t", path)

from pathlib import Path

data_path_file = os.getcwd()
txt = Path(data_path_file + '/' + 'path.txt').read_text()
txt = txt.replace('\n', '')
print ("path_path_file = \t", data_path_file)
print ("path_path_file = \t", txt)

data_path = txt #'/bettik/PROJECTS/pr-geodynamo-ai/sharmam/results/900_samples_r_32_64_t_0_32_p_256/'


file_path = os.path.join(path, 'model_unet')#'/bettik/PROJECTS/pr-geodynamo-ai/sharmam/code/Unet/run_4_may_2024/f_uu/t_comp/notebooks/model_unet'

print ("file_path = \t", file_path)


#if torch.cuda.is_available():
#    map_location=lambda storage, loc: storage.cuda()
#else:
#    map_location='cpu'
    
#print ('Using device:',device)

# ~ ## import testing data

#print ("File in the Manohar folder")

#data_path = '/bettik/PROJECTS/pr-geodynamo-ai/sharmam/results/900_samples_r_32_64_t_0_32_p_256/'
#file_path = os.path.join(path, 'model_unet')#'/bettik/PROJECTS/pr-geodynamo-ai/sharmam/code/Unet/run_4_may_2024/f_uu/t_comp/notebooks/model_unet'

#print ("file_path = \t", file_path)

#data_path = '/bettik/PROJECTS/pr-geodynamo-ai/sharmam/results/900_samples_r_32_64_t_0_32_p_256/'
#file_path = '/bettik/PROJECTS/pr-geodynamo-ai/sharmam/code/Unet/run_4_may_2024/f_uu/r_comp/final/with_nlin_T/notebooks/model_unet'


my_list = os.listdir(file_path)

directory = []

for dire in my_list:
	if dire[-3:]!='npy':
		directory.append(dire)

print ("dire = \t",directory)


mean_input   	= np.load(file_path + '/' + 'mean_input.npy')
std_input 	    = np.load(file_path + '/' + 'std_input.npy')
mean_output 	= np.load(file_path + '/' + 'mean_output.npy')
std_output 	    = np.load(file_path + '/' + 'std_output.npy')

print ("mean_input =\t",mean_input)
print ("mean_output =\t",mean_output)
	
#name = ['UU_r']

train_standarize = Standarization(
		device = device,
        path = data_path,
        samples = 1,
        established =0,
        mode = 'val',
        size1 = 32,
        size2 = 32,
        size3 = 256,
        mean_input  = mean_input,
        std_input   = std_input,
        mean_output = mean_output,
        std_output  = std_output,
        x = ['U_r','U_t','U_p','B_r','B_t','B_p', 'DV1_1', 'DV1_2', 'DV1_3', 'DV2_1','DV2_2', 'DV2_3', 'DV3_1', 'DV3_2', 'DV3_3', 'DB1_1', 'DB1_2', 'DB1_3', 'DB2_1','DB2_2', 'DB2_3', 'DB3_1', 'DB3_2', 'DB3_3','T_r', 'DT_1', 'DT_2', 'DT_3','UUtr_r', 'UUtr_t', 'UUtr_t', 'UBtr_r','UBtr_t','UBtr_p','BUtr_r','BUtr_t','BUtr_p','BBtr_r','BBtr_t','BBtr_p','UTt_r'],
        y = ['UU_r']
			)

train_input_data  = train_standarize.inputs
train_output_data = train_standarize.labels

# ~ print ("data_labels =\t", train_output_data)


train_dataset = Dataset( device, train_input_data, train_output_data )

from blocks.Unet_3D import Unet

UNet = Unet(name='UNet')

UNet.load_state_dict(torch.load(file_path + '/weights.pyt', map_location=map_location))
UNet.eval()

preds, mse_eval = evaluate(dataset=train_dataset, models = UNet)

# ~ print ("")


filename_p = 'preds_train' + '.npy'
filename_d = 'label_train' + '.npy'
np.save(file_path + '/' + filename_p, preds)
np.save(file_path + '/' + filename_d, train_dataset.labels)
	
	
