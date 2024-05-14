from pylab import *
from numpy import *
import sys
import time
import os
import glob
from non_linear_term_function import*

sys.path.append('/nfs_scratch/sharmam/Nonlinear_term/pizza/pizza-master/python/')

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

    my_list_temp = sorted(glob.glob(data_path + '/' + 'frame_us_*.E1e8Ra5e12*'))
    my_list_temp = [i.split('.')[-1] for i in my_list_temp]
    my_list_temp = set(my_list_temp)

    samples = 0

    for name in my_list_temp:
        number_files = sorted(glob.glob(data_path + '/' + 'frame_us_*.%s'%(name)))
        samples = len(number_files)
    
        My_dir_1 = make_dir(save_path, name)

        #print ("My_dir_1",My_dir_1)

        for i in range(1, samples+1):
            f = PizzaFields(tag=name, ivar= i, datadir = data_path)

            r = f.radius
            nphi = f.n_phi_max
            theta = np.linspace(0, 2*np.pi, nphi)
            mmax = f.m_max
            n_m_max = f.n_m_max

            ur = f.us
            ut = f.uphi
            utemp = f.temp

            ur_m = f.us_m
            ut_m = f.uphi_m
            utemp_m = f.temp_m

            utr_r_m = truncate_field(ur_m, mmax, mcut)
            utr_t_m = truncate_field(ut_m, mmax, mcut)
            Ttr_t_m = truncate_field(utemp_m, mmax, mcut)

            utr_r = spec_spat(utr_r_m, nphi)
            utr_t = spec_spat(utr_t_m, nphi)
            Ttr_t = spec_spat(Ttr_t_m, nphi)

            time_stamp = np.round(f.time,12)
            time_stamp = int(time_stamp*1e14)
            number_str = str(time_stamp)
            zero_filled_number = number_str.zfill(14)

            print ("time_stamp = \t",time_stamp)
            print ("nhi = \t",nphi)

            print ("sample = \t",name)

            MYDIR = ("{0:}/{1:}".format(My_dir_1, zero_filled_number))
            CHECK_FOLDER = os.path.isdir(MYDIR)

            if not CHECK_FOLDER:
                os.makedirs(MYDIR)
                print("created folder : ", MYDIR)
            else:
                print(MYDIR, "folder already exists.")

            snapshot_nonline_computation(ur, ut, utemp, utr_r, utr_t, Ttr_t, r, theta, mmax=mmax, mcut=mcut, nphi = nphi, n_m_max= n_m_max, save_path = MYDIR, time_stamp=time_stamp)

    return 0
	

if __name__ == '__main__':
	mcut = 128
	data_path = '/nfs_scratch/schaeffn/data_pizza/E1e8Ra5e12T2D/'
	save_path = '/nfs_scratch/sharmam/Nonlinear_term/results/pizza/full_data_size/128/'
	compute_nonlinear_term(data_path = data_path, save_path = save_path, mcut = mcut)
