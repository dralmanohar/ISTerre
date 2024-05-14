import numpy as np
import h5py

# ~ def load_numpy(filename):
	
	# ~ filename = filename
	
	# ~ data = np.load(filename, allow_pickle=True)
	
	#print (data)
	# ~ r = data.item().get('r')
	# ~ theta = data.item().get('theta')
	# ~ phi = data.item().get('phi')
	# ~ Ur = data.item().get('ur')
	# ~ Ut = data.item().get('ut')
	# ~ Up = data.item().get('up')
	# ~ Br = data.item().get('br')
	# ~ Bt = data.item().get('bt')
	# ~ Bp = data.item().get('bp')
	
	# ~ return r, theta, phi, Ur, Ut, Up, Br, Bt, Bp

def load_numpy_spec(filename):
	
	filename = filename
	
	data = np.load(filename, allow_pickle=True)
	
	# ~ print (data)
	r = data.item().get('r')
	Er = data.item().get('Er')
	Et = data.item().get('Et')
	Ep = data.item().get('Ep')
	Eut = data.item().get('Eut')
	
	return r, Er, Et, Ep, Eut

def load_numpy_scalar(filename):
	
	filename = filename
	
	data = np.load(filename, allow_pickle=True)

	r = data.item().get('r')
	theta = data.item().get('theta')
	phi = data.item().get('phi')				## for the current files
	Tr = data.item().get('Tr')  				## field  Ur
	Ttr = data.item().get('Trr')  				## field  Utheta
	
	return r, theta, phi, Tr, Ttr


def save_npy_scalar(r, theta, phi, Tr, Ttrr, A=None):
	
	var_dict = {'r':r,'theta': theta, 'phi': phi, 'Tr': Tr, 'Trr': Ttrr}
	data_dic = {}
	
	for name, idx in var_dict.items():
		data_dic[name] = idx
	
	filename = A 
	np.save((filename +'.npy'), data_dic)
	
	return 0

def save_npy(r, theta, phi, ur, ut, up, br, bt, bp, A=None):
	
	var_dict = {'r':r,'theta': theta, 'phi': phi, 'ur': ur, 'ut': ut, 'up': up, 'br': br, 'bt': bt, 'bp': bp }
	data_dic = {}
	
	for name, idx in var_dict.items():
		data_dic[name] = idx
	
	# ~ print ('A =\t',A)
	filename = A 
	np.save((filename +'.npy'), data_dic)
	
	return 0
    

def read2D(string1):
		
	path1 = string1
	data = path1.split("/")[-1]
	data1 = data.strip(".h5")

	file_P1_read = h5py.File(path1,'r')
	Pr = file_P1_read[data1]
	
	return Pr

def write_2D(string1, A):
	A = A
	path = string1
	data = path.split("/")[-1]
	data1 = data.strip(".h5")
	
	file_write = h5py.File(path,'w')
	file_write[data1] = A[:,:]
	file_write.close()
	
	return 0
	

def read3D(string1):
		
	path1 = string1
	data = path1.split("/")[-1]
	data1 = data.strip(".h5")

	file_P1_read = h5py.File(path1,'r')
	Pr = file_P1_read[data1]
	
	return Pr


def write_3D(string1, A):
	A = A
	path = string1
	data = path.split("/")[-1]
	data1 = data.strip(".h5")
	
	file_write = h5py.File(path,'w')
	file_write[data1] = A[:,:,:]
	file_write.close()
	
	return 0
