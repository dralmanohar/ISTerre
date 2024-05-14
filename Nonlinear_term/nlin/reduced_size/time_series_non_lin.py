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

def compute_nonline_machine_learning(data_path=None, file_save=None, lcut=None, mcut=None):

    my_list = sorted(os.listdir(data_path))
    number_of_elements = len(my_list)
    no_sample = int(number_of_elements/3)

    f = open("time_data.d",'w')
    f1 = open("filenamew.d", 'w')

    for i in range(no_sample):
        B = my_list[i]
        T = my_list[i +   no_sample]
        U = my_list[i + 2*no_sample]

        field = [B, T, U]

        time_stamp = U.split("_")[1].split(".")[0]
        MYDIR = ("{0:}/{1:}".format(file_save, time_stamp))
        CHECK_FOLDER = os.path.isdir(MYDIR)

        f.write("%f \n"%(float(time_stamp)))

        f1.write("%s \t %s \t %s \n"%(B, T, U))

        if not CHECK_FOLDER:
            os.makedirs(MYDIR)
            print("created folder : ", MYDIR)

        else:
            print(MYDIR, "folder already exists.")

        nlin1, nlin2, C, time, para, D = snapshot_nonline_computation(U, B, T, lcut=lcut, mcut=mcut, path = data_path, save_path = MYDIR, time_stamp=time_stamp)

        comput_truncated_nonlinear(nlin1, nlin2, C, D, para, lcut=lcut, mcut=mcut, data_path=data_path, save_path=MYDIR, time_stamp=time_stamp)

    f.close()
    f1.close()
    return 0

if __name__ == '__main__':

    data_path = '/nfs_scratch/sharmam/Nonlinear_term/data/data_manohar/sample_2000/samples_204/'
    file_save = '/nfs_scratch/sharmam/Nonlinear_term/results/3D_nathanael/reduced_data_for_run/'

    lcut = 50
    mcut = 50
    compute_nonline_machine_learning(data_path=data_path, file_save=file_save, lcut=lcut, mcut=mcut)


