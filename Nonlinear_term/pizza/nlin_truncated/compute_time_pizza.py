from pylab import *
from numpy import *
import sys
import time
import os
import glob
#from non_linear_term_function import*

sys.path.append('/nfs_scratch/sharmam/Nonlinear_term/pizza/pizza-master/python/')

#export PYTHONPATH=$PYTHONPATH:/nfs_scratch/sharmam/Nonlinear_term/code/pizza/pizza-master/python/

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

    my_list_l = []
    my_list_u = []

    for name in my_list_temp:
        if name[-1].isupper():
            my_list_u.append(name)
        else:
            my_list_l.append(name)

    my_list_final = sorted(my_list_l) + sorted(my_list_u)

    del my_list_final[0]

    f1 = open('time.d','w')

    for name in my_list_final:
        number_files = sorted(glob.glob(data_path + '/' + 'frame_us_*.%s'%(name)))

        samples = len(number_files)

        #print ("samples = \t", samples)

        for i in range(1, samples+1):
            f = PizzaFields(tag=name, ivar= i, datadir = data_path)
            time_stamp = np.round(f.time,12)
            f1.write('%1.12f \n'%(time_stamp))
            print ("time_stamp = \t",time_stamp)
    f1.close()
    return 0
	

if __name__ == '__main__':
	mcut = 256
	data_path = '/nfs_scratch/schaeffn/data_pizza/E1e8Ra5e12T2D/'
	save_path = '/nfs_scratch/sharmam/Nonlinear_term/results/pizza/256/'
	compute_nonlinear_term(data_path = data_path, save_path = save_path, mcut = mcut)
