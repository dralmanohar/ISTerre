import sys
import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

sys.path.append('../src')

from dataset import*
from train import*
#from blocks.MS_model import*



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ('Using device:', device)

data_path = '/bettik/PROJECTS/pr-geodynamo-ai/sharmam/results/900_samples_r_32_64_t_0_32_p_256'


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
 


name = ['UU_r']#,'UU_t','UU_p','BB_r','BB_t','BB_p', 'UB_r', 'UB_t', 'UB_p', 'BU_r', 'BU_t', 'BU_p']

print ("Manohar")

dataset = Standarization(
        device = device,
        path = data_path,
        samples = 800,
        established =0,
        mode = 'train',
        size1 = 32,
        size2 = 32,
        size3 = 256,
        #rot = 'rotation',
        #nrot = 4,
        #noise = 'Noise',
        x = ['U_r','U_t','U_p','B_r','B_t','B_p', 'DV1_1', 'DV1_2', 'DV1_3', 'DV2_1','DV2_2', 'DV2_3', 'DV3_1', 'DV3_2', 'DV3_3', 'DB1_1', 'DB1_2', 'DB1_3', 'DB2_1','DB2_2', 'DB2_3', 'DB3_1', 'DB3_2', 'DB3_3','T_r', 'DT_1', 'DT_2', 'DT_3','UUtr_r', 'UUtr_t', 'UUtr_t', 'UBtr_r','UBtr_t','UBtr_p','BUtr_r','BUtr_t','BUtr_p','BBtr_r','BBtr_t','BBtr_p','UTt_r'],
        y = ['UU_r']
        )
						
input_data  =  dataset.inputs
labels_data =  dataset.labels

mean_input   =  dataset.mean_features
std_input    =  dataset.std_features
mean_output  =  dataset.mean_labels
std_output   =  dataset.std_labels

print ("mean_input = \t",mean_input)
print ("mean_input = \t",input_data.shape)

np.save('model_unet/mean_input.npy',mean_input)
np.save('model_unet/std_input.npy' ,std_input)
np.save('model_unet/mean_output.npy',mean_output)
np.save('model_unet/std_output.npy',std_output)

test_dataset = Standarization(
        device = device,
        path = data_path,
        samples = 88,
        established =0,
        mode = 'test',
        size1 = 32,
        size2 = 32,
        size3 = 256,
        #rot = 'rotation',
        #nrot = 4,
        #noise = 'Noise',
        mean_input  = mean_input,
        std_input   = std_input,
        mean_output = mean_output,
        std_output  = std_output,
        x = ['U_r','U_t','U_p','B_r','B_t','B_p', 'DV1_1', 'DV1_2', 'DV1_3', 'DV2_1','DV2_2', 'DV2_3', 'DV3_1', 'DV3_2', 'DV3_3', 'DB1_1', 'DB1_2', 'DB1_3', 'DB2_1','DB2_2', 'DB2_3', 'DB3_1', 'DB3_2', 'DB3_3','T_r', 'DT_1', 'DT_2', 'DT_3','UUtr_r', 'UUtr_t', 'UUtr_t', 'UBtr_r','UBtr_t','UBtr_p','BUtr_r','BUtr_t','BUtr_p','BBtr_r','BBtr_t','BBtr_p','UTt_r'],
        y = ['UU_r']
        )

input_test  =  test_dataset.inputs
labels_test =  test_dataset.labels

#print ("input test = \t",input_test)

train_dataset  = Dataset(device,input_data, labels_data)
valid_dataset  = Dataset(device,input_test, labels_test)	


from blocks.Unet_3D import Unet
UNet = Unet(name='UNet').to(device)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 8, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = 8, shuffle = False)

opti = torch.optim.Adam(UNet.parameters(), lr = 8e-5, weight_decay = 8e-5)#, betas = (0.9, 0.999), eps = 1e-8, amsgrad = True )

rate = torch.optim.lr_scheduler.StepLR(opti, step_size=1000, gamma=0.2)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

print ("count parameters = \t", count_parameters(UNet))
			


loop(
        net=UNet,
        model_path = './model_unet/',
        train_loader = train_loader,
        valid_loader = valid_loader,
        opti = opti,
        rate = rate,
        mean = mean_output,
        std  = std_output,
        epochs = 180
        )


