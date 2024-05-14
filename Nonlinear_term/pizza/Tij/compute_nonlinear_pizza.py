from pylab import *
from numpy import *
import sys
import time
import os
import glob

from Tij import*

sys.path.append('../../pizza/pizza-master/python/')

from pizza import *


################

def truncate_field(A, mmax, mcut):
	
	B = np.zeros(A.shape, dtype = complex)
	
	for i in range(A.shape[1]):
		B[:mcut, i] = A[:mcut,i]
	return B
	

def make_dir(save_path = None, time=None):
	
	MYDIR = ("{0:}/{1:}".format(save_path, time))
	CHECK_FOLDER = os.path.isdir(MYDIR)
	
	if not CHECK_FOLDER:
		os.makedirs(MYDIR)
		print("created folder : ", MYDIR)
	else:
		print(MYDIR, "folder already exists.")
		
	return MYDIR

def compute_nonlinear_term(data_path = None, save_path = None, mcut = None):
	
	my_list = sorted(os.listdir(data_path))
	
	number_of_elements = len(my_list)
	no_sample = int(number_of_elements/3)
	
	for i in range(1,no_sample+1):
		
		f = PizzaFields(tag='E1e8Ra5e12K', ivar= i, datadir = data_path)
		
		r = f.radius
		nphi = f.n_phi_max
		theta = np.linspace(0, 2*np.pi, nphi)
		mmax = f.m_max
					
		ur = f.us
		ut = f.uphi
		utemp = f.temp
		
		ur_m = f.us_m
		ut_m = f.uphi_m
		utemp_m = f.temp_m

		utr_r = truncate_field(ur_m, mmax, mcut)
		utr_t = truncate_field(ut_m, mmax, mcut)
		Ttr_t = truncate_field(utemp_m, mmax, mcut)
		
		time_stamp = f.time
		
		snapshot_nonline_computation(ur, ut, utemp, utr_r, utr_t, Ttr_t, r, theta, mmax=mmax, mcut=mcut, save_path = save_path, time_stamp=time_stamp):
			
	return 0
	

if __name__ == '__main__':
	mcut = 1024
	data_path = '/media/sharmam/Transcend/Pizza_2D_data/'
	save_path = '/media/sharmam/Transcend/Pizza_nonlinear_term/'
	compute_nonlinear_term(data_path = data_path, save_path = save_path, mcut = mcut)
