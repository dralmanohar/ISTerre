from pylab import *
from numpy import *
from pyxshells import *
import shtns
# ~ import xsplot
import sys
from derivative_spherical import *
import time
import os
import glob
from non_linear_term_function import*
from nonlinear_F_AA_AB import*

############ defined sh

lmax = 79
mmax = 63

def compute_nonline_machine_learning(data_path=None, file_save=None, lcut=None, mcut=None):
	
	my_list = sorted(os.listdir(data_path))
	
	number_of_elements = len(my_list)
		
	no_sample = int(number_of_elements/3)
	
	f = open("time_data.d",'w')
		
	for i in range(no_sample):
		
		B = my_list[i]
		T = my_list[i +   no_sample]
		U = my_list[i + 2*no_sample]
		
		# ~ print ("filename = \t",U)
		field = [B, T, U]
		
		time_stamp = U.split("_")[1].split(".")[0]
			
		MYDIR = ("{0:}/{1:}".format(file_save, time_stamp))
		CHECK_FOLDER = os.path.isdir(MYDIR)
		
		# ~ #print ("T = \t",T)
		
		# If folder doesn't exist, then create it.
		if not CHECK_FOLDER:
			os.makedirs(MYDIR)
			print("created folder : ", MYDIR)
		
		else:
			print(MYDIR, "folder already exists.")
			
		nlin1, nlin2, C, time = snapshot_nonline_computation(U, B, T, lcut=lcut, mcut=mcut, path = data_path, save_path = MYDIR, time_stamp=time_stamp)
		
		comput_truncated_nonlinear(nlin1, nlin2, C, lcut=lcut, mcut=mcut, data_path=data_path, save_path=MYDIR, time_stamp=time_stamp)
		
		print ("Time stamp in data =\t",time)
		f.write("%f \n"%(time))
	f.close()
		
	return 0


if __name__ == '__main__':
	
	data_path = '../../data/205_time_frame/'
	file_save = '../../result/205_l_70_m_55/'
	
	lcut = 74
	mcut = 58
	compute_nonline_machine_learning(data_path=data_path, file_save=file_save, lcut=lcut, mcut=mcut)


