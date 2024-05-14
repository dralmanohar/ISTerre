import numpy as np
import os
import h5py

from read_write import*
import glob
######### filenames

def data_generator_frame(path):
	my_list = sorted(os.listdir(data_path))
	sample = 0
	
	n = 192
	
	number_field = len(my_list)*n
	
	train = int(number_field*0.7) 
	test  = int(number_field*0.2) 
	val   = int(number_field*0.1) 
	
	for folder in (my_list):
		path = data_path + folder
		
		########### field files load
		
		file_name1 = path + '/' + 'UW_trunc_%s.npy'%(folder)
		file_name2 = path + '/' + 'BJ_trunc_%s.npy'%(folder)

		
		################## reading files
		######### fields
		
		r, theta, phi, Ur, Ut, Up, _, _, _ = load_numpy(file_name1)          #### velocity field
		_, _, _,       Br, Bt, Bp, _, _, _ = load_numpy(file_name2)          #### Magnetic field

		################ fields		

		for i in range(n):
			sample +=1
			
			if sample==195:
				print ("field component = \t", Br[:,:,i])
				
	return 0


def standrization_test(path):
	my_list = sorted(os.listdir(data_path))
	sample = 0
	
	n = 192

	number_field = len(my_list)*n
	
	train = int(number_field*0.7) 
	test  = int(number_field*0.2) 
	val   = int(number_field*0.1)
	
	data = np.zeros((train, 1, 100, 120))
	data_v = np.zeros((test, 1, 100, 120))
	data_stand = np.zeros((train, 1, 100, 120))
	data_stand_v = np.zeros((test, 1, 100, 120))
	
	mean_field = []
	std_field = []
	
	for folder in (my_list):
		path = data_path + folder
		
		########### field files load
		
		file_name1 = path + '/' + 'UW_trunc_%s.npy'%(folder)
		file_name2 = path + '/' + 'BJ_trunc_%s.npy'%(folder)
		file_name3 = path + '/' + 'tilde_full_AA_%s.npy'%(folder)

		
		################## reading files
		######### fields
		
		r, theta, phi, Ur, Ut, Up, _, _, _ = load_numpy(file_name1)          #### velocity field
		_, _, _,       Br, Bt, Bp, _, _, _ = load_numpy(file_name2)          #### Magnetic field
		_, _, _, UUr, UUt, UUp, BBr, BBt, BBp = load_numpy(file_name3)
		################ fields		
		
		for i in range(n):
			
			# ~ print ("sample = \t", sample)
			if sample<train:
				mean = np.mean(UUr[:,:,i])
				std  = np.std(UUr[:,:,i])
				mean_field.append(mean)
				data[sample,:,:,:] = UUr[:,:,i]
			elif train<=sample<train + test:
				data_v[sample - train,:,:,:] = UUr[:,:,i]	
			sample +=1
		
	
	# ~ print ("validation data test = \t",data[267, :, :, :])
	
	########################################
	
	mean_field = np.mean(np.asarray(mean_field))
	
	#########################################
	for i in range(data.shape[1]):
		
		count = 0
		for j in range(data.shape[0]):
			for k in range(data.shape[2]):
				for l in range(data.shape[3]):
					std += (data[j, i, k, l] - mean_field)**2
					
					count +=1
					
	std_f = np.sqrt(std/count)
	
	# ~ print ("data mean = \t", mean_field)
	# ~ print ("data std = \t", std_f)
	###### standrization of training data
	for i in range(data.shape[1]):
		for j in range(data.shape[0]):
			data_stand[j,i,:,:] = ((data[j,i,:,:] - mean_field)/ (std_f)) 			
	
	
	
	# ~ print ("validation data test = \t",data_stand[267, :, :, :])

	# ~ print ("standrized data = \t", np.mean(data_stand[267,:,:,:]))
	# ~ print ("standrized data = \t", np.std(data_stand[267,:,:,:]))
	# ~ print ("data shape test = \t", data_v.shape)
	######## standrization fo validation data
	
	for i in range(data_v.shape[1]):
		for j in range(data_v.shape[0]):
			data_stand_v[j,i,:,:] = ((data_v[j,i,:,:] - mean_field)/ (std_f)) 			
	
	print ("validation data standrized test = \t",data_stand_v[75, :, :, :])
	
	return 0


##############

if __name__ == '__main__':
	data_path = '../../result/2_time_frame/'
	# ~ data_path = '../../result/test/'
	# ~ data_generator_frame(data_path)
	standrization_test(data_path)
