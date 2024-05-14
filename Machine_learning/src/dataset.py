import torch
import os
from numpy import*
import h5py

import matplotlib
import matplotlib.pyplot as plt

import glob

#from skimage.util import random_noise

plt.rcParams.update({'mathtext.fontset':'cm'})


def shift_image(data, n_shift):

    new_data = torch.zeros(data.shape)
    n_phi = data.shape[2]
    
    for k in range(0, n_shift):
        new_data[:, :, k] = torch.from_numpy(data[:,:, n_phi - n_shift + k])

    for k in range(0, n_phi - n_shift):
        new_data[:, :, n_shift + k] = torch.from_numpy(data[:, :, k])

    return new_data
	

class Standarization(torch.utils.data.Dataset):
    def __init__(self, device, path, samples, size1, size2, size3, x, y, established=0, mode=0, mean_input = 0, std_input = 0, mean_output = 0, std_output = 0, eval_folder=0, norm = 0, max_input = 0, min_input = 0, max_output = 0, min_output = 0, num_deri = 0, num_scalar = 0, num_scalar_deri = 0, stat = 0, noise = None, rot = None, nrot = None):

        self.device = device
        self.path = path
        self.samples = samples - established
        self.size1 = size1
        self.size2 = size2
        self.size3 = size3
        self.established = established
        self.mode = mode
        self.x = x
        self.y = y
        self.eval_folder = eval_folder
        self.mean = mean
        self.std = std
        self.norm = norm


        self.mean_features  = torch.zeros(len(x), dtype=torch.float32)
        self.std_features   = torch.zeros(len(x), dtype=torch.float32)

        self.mean_labels 	= torch.zeros(len(y), dtype=torch.float32)
        self.std_labels  	= torch.zeros(len(y), dtype=torch.float32)


        if rot=='rotation':

            self.inputs_wt_rotation = torch.zeros(self.samples, len(x), size1, size2, size3, dtype=torch.float32)
            self.labels_wt_rotation = torch.zeros(self.samples, len(y), size1, size2, size3, dtype=torch.float32)

            self.inputs_rotation = torch.zeros(self.samples*nrot, len(x), size1, size2, size3, dtype=torch.float32)
            self.labels_rotation = torch.zeros(self.samples*nrot, len(y), size1, size2, size3, dtype=torch.float32)

            self.inputs = torch.zeros(self.samples + self.samples*nrot, len(x), size1, size2, size3, dtype=torch.float32)
            self.labels = torch.zeros(self.samples + self.samples*nrot, len(y), size1, size2, size3, dtype=torch.float32)

            if noise=='Noise':

                self.inputs_norm_wt_noise = torch.zeros(self.samples + self.samples*nrot, len(x), size1, size2, size3, dtype=torch.float32)
                self.labels_norm_wt_noise = torch.zeros(self.samples + self.samples*nrot, len(x), size1, size2, size3, dtype=torch.float32)

                self.inputs_norm_noise = torch.zeros(self.samples + self.samples*nrot, len(x), size1, size2, size3, dtype=torch.float32)
                self.labels_norm_noise = torch.zeros(self.samples + self.samples*nrot, len(y), size1, size2, size3, dtype=torch.float32)

                self.inputs_norm = torch.zeros(2*self.samples*nrot + 2*self.samples, len(x), size1, size2, size3, dtype=torch.float32)
                self.labels_norm = torch.zeros(2*self.samples*nrot + 2*self.samples, len(y), size1, size2, size3, dtype=torch.float32)
            
            else:
                self.inputs_norm = torch.zeros(self.samples + self.samples*nrot, len(x), size1, size2, size3, dtype=torch.float32)
                self.labels_norm = torch.zeros(self.samples + self.samples*nrot, len(y), size1, size2, size3, dtype=torch.float32)

        elif noise=='Noise':

            self.inputs = torch.zeros(self.samples, len(x), size1, size2, size3, dtype=torch.float32)
            self.labels = torch.zeros(self.samples, len(y), size1, size2, size3, dtype=torch.float32)

            self.inputs_norm_wt_noise = torch.zeros(self.samples, len(x), size1, size2, size3, dtype=torch.float32)
            self.labels_norm_wt_noise = torch.zeros(self.samples, len(y), size1, size2, size3, dtype=torch.float32)

            self.inputs_norm_noise = torch.zeros(self.samples, len(x), size1, size2, size3, dtype=torch.float32)
            self.labels_norm_noise = torch.zeros(self.samples, len(y), size1, size2, size3, dtype=torch.float32)

            self.inputs_norm = torch.zeros(2*self.samples, len(x), size1, size2, size3, dtype=torch.float32)
            self.labels_norm = torch.zeros(2*self.samples, len(y), size1, size2, size3, dtype=torch.float32)

        else:

            self.inputs = torch.zeros(self.samples, len(x), size1, size2, size3, dtype=torch.float32)
            self.labels = torch.zeros(self.samples, len(y), size1, size2, size3, dtype=torch.float32)

#            self.inputs_norm = torch.zeros(self.samples, len(x), size1, size2, size3, dtype=torch.float32)
#            self.labels_norm = torch.zeros(self.samples, len(y), size1, size2, size3, dtype=torch.float32)


        #### data input
        pathf = path  + '/' + mode

        for i, f in enumerate(x):
            for j, c in enumerate(sorted(glob.glob(pathf + '/' + str(f) + '_*'))):
                if j < established:
                    continue
                elif j >= samples:
                    break

                p = j - established
                file = h5py.File(c, 'r')
                #dset = file[c.split("/")[-1].strip(".h5")][:,32-16:32+16, :]
                dset = file[c.split("/")[-1].strip(".h5")][:, :, :]

                #print ("shape = dset", dset[:,:,:].shape)

                if rot=='rotation':
                    self.inputs_wt_rotation[p][i] = torch.from_numpy(dset[()])

                    n_shift = int(dset.shape[-1]/nrot)
                    for k in range(nrot):
                        self.inputs_rotation[p*nrot + k ][i]    = shift_image(dset, n_shift*k) #torch.from_numpy(dset[()]) + torch.randn(dset.shape)
                else:
                    self.inputs[p][i] = torch.from_numpy(dset[()])

        if rot=='rotation':
            self.inputs                = torch.cat([self.inputs_wt_rotation, self.inputs_rotation], dim = 0)


        ########### standrization

        if mode=='train':

            self.mean_features = torch.mean(self.inputs, dim = [0, 2, 3, 4])
            self.std_features  = torch.std(self.inputs,  dim = [0, 2, 3, 4])

            for i in range(0,self.inputs.shape[0]):
                for j in range(0, self.inputs.shape[1]):
                    if noise == 'Noise':
                        self.inputs_norm_wt_noise[i][j] = ((self.inputs[i][j] - self.mean_features[j])/(self.std_features[j]))
                        self.inputs_norm_noise[i][j]    = self.inputs_norm_wt_noise[i][j] + torch.randn(self.inputs_norm_wt_noise[i][j].shape)
                    elif rot=='rotation'  and noise == 'Noise':
                        self.inputs_norm_wt_noise[i][j] = ((self.inputs[i][j] - self.mean_features[j])/(self.std_features[j]))
                        self.inputs_norm_noise[i][j]    = self.inputs_norm_wt_noise[i][j] + torch.randn(self.inputs_norm_wt_noise[i][j].shape)
                    else:
                        self.inputs[i][j] = ((self.inputs[i][j] - self.mean_features[j])/(self.std_features[j]))

            if noise=='Noise':
                self.inputs_norm = torch.cat([self.inputs_norm_wt_noise, self.inputs_norm_noise], dim = 0)
            elif rot == 'rotation' and noise=='Noise':
                self.inputs_norm = torch.cat([self.inputs_norm_wt_noise, self.inputs_norm_noise], dim = 0)

        else:

            for i in range(0,self.inputs.shape[0]):
                for j in range(0, self.inputs.shape[1]):

                    if noise == 'Noise':
                        self.inputs_norm_wt_noise[i][j] = ((self.inputs[i][j] - mean_input[j])/(std_input[j]))
                        #self.inputs_norm_noise[i][j]    = self.inputs_norm_wt_noise[i][j] + torch.randn(self.inputs_norm_wt_noise[i][j].shape)
                    elif rot=='rotation'  and noise == 'Noise':
                        self.inputs_norm_wt_noise[i][j] = ((self.inputs[i][j] - mean_input[j])/(std_input[j]))
                        #self.inputs_norm_noise[i][j]    = self.inputs_norm_wt_noise[i][j] + torch.randn(self.inputs_norm_wt_noise[i][j].shape)
                    else:
                        self.inputs[i][j] = ((self.inputs[i][j] - mean_input[j])/(std_input[j]))

            if noise=='Noise':
                self.inputs_norm = self.inputs_norm_wt_noise #torch.cat([self.inputs_norm_wt_noise, self.inputs_norm_noise], dim = 0)
            elif rot == 'rotation' and noise=='Noise':
                self.inputs_norm = self.inputs_norm_wt_noise[i][j] #torch.cat([self.inputs_norm_wt_noise, self.inputs_norm_noise], dim = 0)


        ### data output
        pathf = path  + '/' + mode

        for i, f in enumerate(y):
            for j, c in enumerate(sorted(glob.glob(pathf + '/' + str(f) + '_*'))):
                if j < established:
                    continue
                elif j >= samples:
                    break

                p = j - established
                file = h5py.File(c, 'r')
                #dset = file[c.split("/")[-1].strip(".h5")][:, 32-16:32+16, :]
                dset = file[c.split("/")[-1].strip(".h5")][:, :, :]

                if rot=='rotation':
                    self.labels_wt_rotation[p][i] = torch.from_numpy(dset[()])
                    n_shift = int(dset.shape[-1]/nrot)
                    for k in range(nrot):
                        self.labels_rotation[p*nrot + k ][i]    = shift_image(dset, n_shift*k) #torch.from_numpy(dset[()]) + torch.randn(dset.shape)

                else:
                    self.labels[p][i] = torch.from_numpy(dset[()])

        if rot=='rotation':
            self.labels                = torch.cat([self.labels_wt_rotation, self.labels_rotation], dim = 0)
        else:
            self.labels                = self.labels



        ### standrization
        if mode=='train':

            self.mean_labels = torch.mean(self.labels, dim = [0, 2, 3, 4])
            self.std_labels  = torch.std(self.labels,  dim = [0, 2, 3, 4])
            for i in range(0,self.labels.shape[0]):
                for j in range(0, self.labels.shape[1]):
                    if noise == 'Noise':
                        self.labels_norm_wt_noise[i][j] = ((self.labels[i][j] - self.mean_labels[j])/(self.std_labels[j]))
                        self.labels_norm_noise[i][j]    = self.labels_norm_wt_noise[i][j] + torch.randn(self.labels_norm_wt_noise[i][j].shape)

                    elif rot=='rotation' and noise == 'Noise':
                        self.labels_norm_rotation_wt_noise[i][j] = ((self.labels[i][j] - self.mean_labels[j])/(self.std_labels[j]))
                        self.labels_norm_rotation_noise[i][j]    = self.labels_norm_rotation_wt_noise[i][j] + torch.randn(self.labels[i][j].shape)
                    else:
                        self.labels[i][j] = ((self.labels[i][j] - self.mean_labels[j])/(self.std_labels[j]))

            if noise=='Noise':
                print ("Manohar")
                self.labels_norm = torch.cat([self.labels_norm_wt_noise, self.labels_norm_noise], dim = 0)
            elif rot=='rotaion' and noise=='Noise':
                self.labels_norm = torch.cat([self.labels_norm_rotation_wt_noise, self.labels_norm_rotation_noise], dim = 0)

        else:
            
            for i in range(0,self.labels.shape[0]):
                for j in range(0, self.labels.shape[1]):
                    if noise == 'Noise':
                        self.labels_norm_wt_noise[i][j] = ((self.labels[i][j] - mean_output[j])/(std_output[j]))
                       # self.labels_norm_noise[i][j]    = self.labels_norm_wt_noise[i][j] + torch.randn(self.labels[i][j].shape)
                    elif rot=='rotation' and noise == 'Noise':
                        self.labels_norm_rotation_wt_noise[i][j] = ((self.labels[i][j] - mean_output[j])/(std_output[j]))
                        #self.labels_norm_rotation_noise[i][j]    = self.labels_norm_rotation_wt_noise[i][j] + torch.randn(self.labels[i][j].shape)
                    else:
                        self.labels[i][j] = ((self.labels[i][j] - mean_output[j])/(std_output[j]))


            if noise=='Noise':
                self.labels_norm =  self.labels_norm_wt_noise #torch.cat([self.labels_norm_wt_noise, self.labels_norm_noise], dim = 0)
            elif rot=='rotaion' and noise=='Noise':
                self.labels_norm = self.labels_norm_rotation_wt_noise#torch.cat([self.labels_norm_rotation_wt_noise, self.labels_norm_rotation_noise], dim = 0)
						
				
class Dataset(torch.utils.data.Dataset):

    def __init__(self, device, data_input, data_output):

        self.device = device
        self.inputs = data_input
        self.labels = data_output
        self.samples = data_input.shape[0]

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = self.inputs[idx]
        labels = self.labels[idx]

        return (inputs.to(self.device), labels.to(self.device))
