import numpy as np
import os
import h5py

from read_write import*
import glob
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

def data_generator(path):
	my_list = sorted(os.listdir(data_path))
	for folder in (my_list):
		path = data_path + folder
		
		########### field files load
		
		file_name1 = path + '/' + 'UW_trunc_%s.npy'%(folder)
		file_name2 = path + '/' + 'BJ_trunc_%s.npy'%(folder)
		file_name3 = path + '/' + 'TTr_%s.npy'%(folder)
		
		##### derivative files load
		
		file_name4 = path + '/' + 'TDVF_%s.npy'%(folder)
		file_name5 = path + '/' + 'TDBF_%s.npy'%(folder)
		file_name6 = path + '/' + 'TFTr_%s.npy'%(folder)
		
		############## nonlinear terms load
		
		file_name7 = path + '/' + 'overline_tilde_full_AA_%s.npy'%(folder)
		file_name8 = path + '/' + 'overline_tilde_full_AB_%s.npy'%(folder)
		
		
		################## reading files
		######### fields
		
		r, theta, phi, Ur, Ut, Up, _, _, _ = load_numpy(file_name1)          #### velocity field
		_, _, _,       Br, Bt, Bp, _, _, _ = load_numpy(file_name2)          #### Magnetic field
		_, _, _, _, Tr 					   = load_numpy_scalar(file_name3) 
		
		#### derivative loaded
		
		DV11, DV12, DV13, DV21, DV22, DV23, DV31, DV32, DV33 = load_numpy(file_name4) #### nonlinear terms
		
		DB11, DB12, DB13, DB21, DB22, DB23, DB31, DB32, DB33 = load_numpy(file_name5) #### nonlinear terms
		
		_,_,_,_,_,_, DT1, DT2, DT3    = load_numpy(file_name6)
		
		########### nonlinear terms 
		
		_, _, _, UUr, UUt, UUp, BBr, BBt, BBp = load_numpy(file_name7) #### nonlinear terms
		_, _,_, UBr, UBt, UBp, BUr, BUt, BUp  = load_numpy(file_name8) #### nonlinear terms
		
		
		################ fields
		U = {'Ur':Ur, 'Ut': Ut, 'Up':Up}
		B = {'Br':Br, 'Bt': Bt, 'Bp':Bp}
		T = {'Tr':Tr}
		
		############## derivatives
		
		DV = {'DV11':DV11, 'DV12':DV12, 'DV13':DV13, 'DV21':DV21, 'DV22':DV22, 'DV23':DV23, 'DV31':DV31, 'DV32':DV32, 'DV33':DV33}
		
		DB = {'DB11':DB11, 'DB12':DB12, 'DB13':DB13, 'DB21':DB21, 'DB22':DB22, 'DB23':DB23, 'DB31':DB31, 'DB32':DB32, 'DB33':DB33}
		
		DT = {'DT1':DT1, 'DT2':DT2, 'DT3':DT3}
		
		
		############ nonlinear terms
		
		UU = {'UUr':UUr, 'UUt':UUt, 'UUp':UUp}
		BB = {'BBr':BBr, 'BBt':BBt, 'BBp':BBp}
		UB = {'UBr':UBr, 'UBt':UBt, 'UBp':UBp}
		BU = {'BUr':BUr, 'BUt':BUt, 'BUp':BUp}
		
		#### velocity field
		
		
		fields = [U, B, T, DV, DB, DT, UU, BB, UB, BU]#, T]
	
		n = len(phi)
		
		train = int(n*0.7)
		test  = train + int(n*0.25)
		val  = test + int(n*0.05)
		
		for name in fields:
			for keys, values in name.items():
				comp = keys[-1]
				variable = keys[:-1]
				
				for i in range(0, n):
					if i<train:
						path = data_path + folder
						dire = make_direcotry(path, 'train')
					elif train<=i<test:
						path = data_path + folder
						dire = make_direcotry(path, 'test')
					else:
						path = data_path + folder
						dire = make_direcotry(path, 'val')
					filename = dire + '/' + variable + '_{0:}_{1:}'.format(comp, i) + '.h5'
					field = values[:,:,i]
					write_2D(filename, field)
	return 0



def data_generator_frame(path):
	my_list = sorted(os.listdir(data_path))
	sample = 0
	
	n = 192
	
	number_field = len(my_list)*n
	
	train = int(number_field*0.7) 
	test  = int(number_field*0.2) 
	val   = int(number_field*0.1) 
	print ("train = \t",train)		

	
	for folder in (my_list):
		path = data_path + folder
		
		########### field files load
		
		file_name1 = path + '/' + 'UW_trunc_%s.npy'%(folder)
		file_name2 = path + '/' + 'BJ_trunc_%s.npy'%(folder)
		file_name3 = path + '/' + 'TTr_%s.npy'%(folder)
		
		# ~ file_name1 = path + '/' + 'UW%s.npy'%(folder)
		# ~ file_name2 = path + '/' + 'BJ%s.npy'%(folder)
		# ~ file_name3 = path + '/' + 'TTr_%s.npy'%(folder)
		
		##### derivative files load
		
		file_name4 = path + '/' + 'TDVF_%s.npy'%(folder)
		file_name5 = path + '/' + 'TDBF_%s.npy'%(folder)
		file_name6 = path + '/' + 'TFTr_%s.npy'%(folder)
		
		# ~ ############## nonlinear terms load
		
		file_name7 = path + '/' + 'tilde_full_AA_%s.npy'%(folder)
		file_name8 = path + '/' + 'tilde_full_AB_%s.npy'%(folder)
		
		
		# ~ file_name7 = path + '/' + 'AA_%s.npy'%(folder)
		# ~ file_name8 = path + '/' + 'AB_%s.npy'%(folder)
		
		
		################## reading files
		######### fields
		
		r, theta, phi, Ur, Ut, Up, _, _, _ = load_numpy(file_name1)          #### velocity field
		_, _, _,       Br, Bt, Bp, _, _, _ = load_numpy(file_name2)          #### Magnetic field
		_, _, _,_,Tr 					   = load_numpy_scalar(file_name3) 
		
		# ~ #### derivative loaded
		
		DV11, DV12, DV13, DV21, DV22, DV23, DV31, DV32, DV33 = load_numpy(file_name4) #### nonlinear terms
		
		DB11, DB12, DB13, DB21, DB22, DB23, DB31, DB32, DB33 = load_numpy(file_name5) #### nonlinear terms
		
		_,_,_,_,_,_, DT1, DT2, DT3    = load_numpy(file_name6)
		
		# ~ ########### nonlinear terms 
		
		_, _, _, UUr, UUt, UUp, BBr, BBt, BBp = load_numpy(file_name7) #### nonlinear terms
		_, _,_, UBr, UBt, UBp, BUr, BUt, BUp  = load_numpy(file_name8) #### nonlinear terms
		
		
		################ fields
		U = {'Ur':Ur, 'Ut': Ut, 'Up':Up}
		B = {'Br':Br, 'Bt': Bt, 'Bp':Bp}
		T = {'Tr':Tr}
		
		# ~ ############## derivatives
		
		DV = {'DV11':DV11, 'DV12':DV12, 'DV13':DV13, 'DV21':DV21, 'DV22':DV22, 'DV23':DV23, 'DV31':DV31, 'DV32':DV32, 'DV33':DV33}
		
		DB = {'DB11':DB11, 'DB12':DB12, 'DB13':DB13, 'DB21':DB21, 'DB22':DB22, 'DB23':DB23, 'DB31':DB31, 'DB32':DB32, 'DB33':DB33}
		
		DT = {'DT1':DT1, 'DT2':DT2, 'DT3':DT3}
		
		
		############ nonlinear terms
		
		UU = {'UUr':UUr, 'UUt':UUt, 'UUp':UUp}
		BB = {'BBr':BBr, 'BBt':BBt, 'BBp':BBp}
		UB = {'UBr':UBr, 'UBt':UBt, 'UBp':UBp}
		BU = {'BUr':BUr, 'BUt':BUt, 'BUp':BUp}
		
		# ~ #### velocity field
		
		fields = [U, B, T, DV, DB, DT, UU, BB, UB, BU]#, T]
		
		for i in range(0, n):
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

			# ~ ## ~ name_file = sample
			number_str = str(sample)

			zero_filled_number = number_str.zfill(6)
			
			
			for name in fields:
				for keys, values in name.items():
					comp = keys[-1]
					variable = keys[:-1]
			
					filename = dire + '/' + variable + '_{0:}_{1:}'.format(comp, zero_filled_number) + '.h5'
					field = values[26:74,36:84,i]
					# ~ ## ~ print ("shape filed =\t", field.shape)
					write_2D(filename, field)
					
	return 0


def standarization():
	StandField = {}

	for name in fields:
		for keys, values in name.items():
			
			values = (values - values.mean())/(values.std())
			
			StandField[keys] = values
		
	train = int(n*0.67)
	test  = train + int(n*0.25)
	val  = test + int(n*0.08)
	
	for keys, values in StandField.items():
		comp = keys[-1]
		variable = keys[:-1]
		# ~ #print ("values =\t",values)
		for i in range(0, n):
			if i<train:
				path = '../../results/data'
				dire = make_direcotry(path, 'train')
			elif train<=i<test:
				path = '../../results/data'
				dire = make_direcotry(path, 'test')
			else:
				path = '../../results/data'
				dire = make_direcotry(path, 'val')
				
			filename = dire + '/' + variable + '_{0:}_{1:}'.format(comp, i) + '.h5'
			field = values[:,:,i]
			write_2D(filename, field)
	return 0

##############

if __name__ == '__main__':
	data_path = '../../result/205_frame_l_50/'
	# ~ data_path = '../../result/test/'
	data_generator_frame(data_path)
