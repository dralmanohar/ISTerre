import numpy as np
import os
import h5py
import glob
from read_write import*
######### filenames

def make_direcotry(path, com):
	MYDIR = ("{0:}/{1:}".format(path, com))
	CHECK_FOLDER = os.path.isdir(MYDIR)
	
	if not CHECK_FOLDER:
		os.makedirs(MYDIR)
		print("created folder : ", MYDIR)
	else:
		print(MYDIR, "folder already exists.")
	
	return MYDIR
	

	var_dict = {'r':r,'theta': theta, 'ur': ur, 'ut': ut, 'temp':temp, 'durr':durr, 'durt': durt, 'dutr': dutr, 'dutt': dutt, 'dTr': dTr, 'dTt': dTt, 'nlinr': nlinr, 'nlint': nlint}
	
def load_numpy(filename):
	
	filename = filename
	
	data = np.load(filename, allow_pickle=True)
	
	# ~ print (data)
	r = data.item().get('r')
	theta = data.item().get('theta')
	ur = data.item().get('ur')
	ut = data.item().get('ut')
	temp = data.item().get('temp')
	durr = data.item().get('durr')
	durt = data.item().get('durt')
	dutr = data.item().get('dutr')
	dutt = data.item().get('dutt')
	dTr = data.item().get('dTr')
	dTt = data.item().get('dTt')
	nlinr = data.item().get('nlinr')
	nlint = data.item().get('nlint')
	nlintr_r = data.item().get('nlintr_r')
	nlintr_t = data.item().get('nlintr_t')
	
	return r, theta, ur, ut, temp, durr, durt, dutr, dutt, dTr, dTt, nlinr, nlint, nlintr_r, nlintr_t


def data_generator_frame(path):
	my_list = sorted(os.listdir(data_path))
	sample = 0
	
	number_field = len(my_list)
	
	train = int(number_field*0.8) 
	test  = int(number_field*0.15) 
	val   = int(number_field*0.05)
	
	print ("train = \t",train)		
	print ("test = \t",test)		
	print ("val = \t",val)		

	
	for folder in (my_list):
		path = data_path + folder
		
		########### field files load

		file_name1 = path + '/' + 'tilde_nlin%s.npy'%(folder)

		##### derivative files load		

		
		r, theta, ur, ut, temp, durr, durt, dutr, dutt, dTr, dTt, nlinr, nlint, nlintr_r, nlintr_t = load_numpy(file_name1)          #### velocity field
		
		# ~ #### derivative loaded

		################ fields

		U = {'ur':ur, 'ut': ut}
		T = {'T':temp}
		
		# ~ ############## derivatives
		
		DV = {'durr':durr, 'durt':durt, 'dutr':dutr, 'dutt':dutt}
		
		DT = {'dTr':dTr, 'dTt':dTt}
		
		
		############ nonlinear terms
		
		UU = {'nlinr':nlinr, 'nlint':nlint,'nlintr_r':nlintr_r, 'nlintr_t':nlintr_t}
		
		# ~ #### velocity field
		
		fields = [U, T, DV, DT, UU]#, T]
		
		print (sample)
		
		if sample<train:
				path = data_path
				dire = make_direcotry(path, 'train')
				sample +=1
		
		elif train<=sample<test + train:
				path = data_path 
				dire = make_direcotry(path, 'test')
				sample +=1
		
		else:
			path = data_path 
			dire = make_direcotry(path, 'val')
			sample +=1

		
		number_str = str(sample)

		zero_filled_number = number_str.zfill(6)
		
		print ("zero_filled_number = \t", zero_filled_number)
		
		count = 0
		for name in fields:
			for keys, values in name.items():
				comp = keys[-1]
				variable = keys[:-1]
						
				filename = dire + '/' + variable + '_{0:}_{1:}'.format(comp, zero_filled_number) + '.h5'
				field = values
				write_2D(filename, field)
					
	return 0

##############

if __name__ == '__main__':
	data_path = '/home/sharmam/Research/ISTerre_Project/Nonlinear_term/result/Pizza_ML_trunc_256/'
	# ~ data_path = '../../result/test/'
	data_generator_frame(data_path)
