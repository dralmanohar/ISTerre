from pylab import *
from numpy import *
import sys
import time
import os
import glob
from non_linear_term_function import*

sys.path.append('../pizza/pizza-master/python/')

from pizza import *


################

def make_dir(save_path = None, time=None):
	
	MYDIR = ("{0:}/{1:}".format(save_path, time))
	CHECK_FOLDER = os.path.isdir(MYDIR)
	
	if not CHECK_FOLDER:
		os.makedirs(MYDIR)
		print("created folder : ", MYDIR)
	else:
		print(MYDIR, "folder already exists.")
		
	return MYDIR

def compute_nonlinear_term(data_path = None, save_path = None):
	
	my_list = sorted(os.listdir(data_path))
	
	number_of_elements = len(my_list)
	no_sample = int(number_of_elements/3)
	
	file1 = open("time.d", 'w')

	for i in range(1,no_sample+1):
		
		f = PizzaFields(tag='E1e8Ra5e12K', ivar= i, datadir = data_path)
		
		r = f.radius
		nphi = f.n_phi_max
		theta = np.linspace(0, 2*np.pi, nphi)
				
		ur = f.us
		ut = f.uphi
		utemp = f.temp
		time_stamp = f.time
		
		print ("ur",ur.shape)
		
		print ("time = \t",time_stamp)
		
		file1.write("%d \t %f \n"%(i, time_stamp))
	file1.close()
		
	return 0
	

if __name__ == '__main__':
	data_path = '/media/sharmam/Transcend/Pizza_2D_data/'
	save_path = '/media/sharmam/Transcend/Pizza_nonlinear_term/'
	compute_nonlinear_term(data_path, save_path)
