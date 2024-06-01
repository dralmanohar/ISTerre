from __future__ import division

import numpy as np
import pandas as pd
from numpy import sin, cos, arccos, sqrt, pi
# ~ import nonlinear_term_calculation*

## test if we are displaying on a screen 
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from scipy import special
from pylab import*
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import legendre
import pylab as pl
import matplotlib.ticker as mtick
from matplotlib.ticker import FixedFormatter
from matplotlib import ticker
import numpy as np
from scipy.integrate import quad
import linecache
from matplotlib import rc, font_manager, patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import datetime
# import pandas for its converter that is then used in pyplot!
import os
from matplotlib import ticker
import matplotlib.ticker as mticker

A = 10

B = 6.5
C = 0.3
D = 4

E = 3.5
#~ E = 3.64
F = 2.5
# ~ F = 2.4
G = 10

lw0 = 0.5
lw11 = 1
lw1 = 1.5
lw2 = 2.5
lw01 = lw0*0.5
ms1 = 2.5
lw75 = 0.75
lw35 = 0.35
lw25 = 0.25
		
font = {'family' : 'serif', 'weight' : 'normal', 'size' : A}
plt.rc('text', usetex=True)

plt.rcParams['xtick.major.size'] = B
plt.rcParams['xtick.major.width'] = C
plt.rcParams['xtick.minor.size'] = D
plt.rcParams['xtick.minor.width'] = C
plt.rcParams['ytick.major.size'] = B
plt.rcParams['ytick.major.width'] = C
plt.rcParams['ytick.minor.size'] = D
plt.rcParams['ytick.minor.width'] = C  

	
	
################ 

path = './'
number_folder = os.listdir(path)

#print (number_folder)


number_folder.remove('To_get_full_field_stack.py')
number_folder.remove('outlog')
number_folder.remove('f_uu')
number_folder.remove('f_ub')
number_folder.remove('f_bu')
number_folder.remove('f_bb')
number_folder.remove('histogram_plot.py')
number_folder.remove('pred_true_distribution.py')
number_folder.remove('spec_comp.py')
number_folder.remove('plot_spectrum.py')
number_folder.remove('plot_in_sphere.py')
number_folder.remove('__pycache__')
number_folder.remove('UB_trunc_0299.npy')
number_folder.remove('plot_nonliner_full_sphere_real_space.py')
number_folder.remove('loss_function_plot.py')

#print (number_folder)

path_all_fields = []

for i in range(len(number_folder)):
    path_sub = os.listdir(number_folder[i])

    path_sub.remove('evaluate_ML.py')
    path_sub.remove('path.txt')
    path_sub.remove('plot_in_sphere.py')
    path_sub.remove('plot_nonliner_real_space.py')
    path_sub.remove('read_write.py')
    path_sub.remove('UB_trunc_0299.npy')
    path_sub.remove('__pycache__')
    path_sub.remove('To_get_full_field_stack.py')
    #path_sub.remove('outlog')
    
    #print ("path_sub = \t", path_sub)

    for sub_dir in path_sub:

        sub_dir_path = path + '/' + number_folder[i] + '/' + sub_dir

        path_field = os.listdir(sub_dir_path)

        #print ("sub_dir = \t", sub_dir_path, "\t path_field = \t",path_field)

        for field_dir  in path_field:

            path_model = sub_dir_path + '/' + field_dir + '/' + 'notebooks' + '/' + 'model_unet'

            path_all_fields.append(path_model)
         #   print ("path field = \t", path_model )

data_path = []
field_comp = []
data_field = []

for file in path_all_fields:

    file_name = file.split("/")

    data_path.append(file_name[2])
    data_field.append(file_name[3])
    field_comp.append(file_name[4])

data_path = np.unique(data_path)
data_field = np.unique(data_field)
field_comp = np.unique(field_comp)

data_path.sort()

print ("data_path = \t", data_path)

path_data_dir = './'

model = 'model_unet'

for field in data_field:
	
    for comp in field_comp:
        csv_path_32 = path_data_dir + '/' + data_path[0] + '/' + field + '/' + comp + '/' +  'notebooks' + '/' + model
        csv_path_64 = path_data_dir + '/' + data_path[1] + '/' + field + '/' + comp + '/' + 'notebooks' + '/' + model

        field_var = field.split("_")[1]
        comp_var  = comp.split("_")[0]

        if comp_var == 'r':
            #print ("Manohar in the different loop")
            comp_var = 'r'
        elif comp_var == 't':
            comp_var = 'theta'
        elif comp_var == 'p':
            comp_var = 'phi'

        #print ("field = \t", field_var, "\t comp = \t", comp_var)

        #print ("csv_path_32 = \t", csv_path_32, "\t csv_path_64 = \t", csv_path_64)

        data_32      = pd.read_csv(csv_path_32 +  '/' + 'losses.csv', names=['T','V','TA1', 'TA2','TA3','VA1', 'VA2', 'VA3','CT','CV'], header=None)

        data_64      = pd.read_csv(csv_path_64 +  '/' + 'losses.csv', names=['T','V','TA1', 'TA2','TA3','VA1', 'VA2', 'VA3','CT','CV'], header=None)

        xd_32     = np.arange(len(data_32['V']))
        xd_64     = np.arange(len(data_64['V']))

        fig, axes = plt.subplots(1,2,figsize=(2*E,F))
        
        axes[0].semilogy(xd_32,   abs(data_32['T']), 'r', lw=lw1, label = r"$T$"'')
        axes[0].semilogy(xd_64,   abs(data_64['T']), 'g', lw=lw1)

        axes[0].semilogy(xd_32,   abs(data_32['V']), '--r', lw=lw1, label = r"$V$")
        axes[0].semilogy(xd_64,   abs(data_64['V']), '--g', lw=lw1)
        
        axes[0].set_xlabel(r"$Epochs$")
        axes[0].set_ylabel(r"$Loss$")
        
        axes[0].legend(bbox_to_anchor = (0.25, 0.4), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 0.8*G, ncol=1)
        fig.tight_layout()

        ##################

        axes[1].semilogy(xd_32,   abs(data_32['CT']), 'r', lw=lw1                ,label = r"$0-32$")
        axes[1].semilogy(xd_64,   abs(data_64['CT']), 'g', lw=lw1              ,label = r"$32-64$")
        
        if comp_var=='r':
            #print ("Manohar in radial field")
            axes[0].set_title(r"$\overline{\tilde{F}_{%s}^%s}$"%(field_var, comp_var))
            axes[1].set_title(r"$\overline{\tilde{F}_{%s}^%s}$"%(field_var, comp_var))
        elif comp_var=='theta':
            axes[0].set_title(r"$\overline{\tilde{F}_{%s}^\%s}$"%(field_var, comp_var))
            axes[1].set_title(r"$\overline{\tilde{F}_{%s}^\%s}$"%(field_var, comp_var))
        elif comp_var =='phi':
            axes[0].set_title(r"$\overline{\tilde{F}_{%s}^\%s}$"%(field_var, comp_var))
            axes[1].set_title(r"$\overline{\tilde{F}_{%s}^\%s}$"%(field_var, comp_var))


        axes[1].set_xlabel(r"$Epochs$")
        axes[1].set_ylabel(r"$C$")

        axes[1].legend(bbox_to_anchor = (0.95, 0.4), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 0.8*G, ncol=2)
        fig.tight_layout()

        path_dir = './'
        MYDIR = path_dir + '/' + '{0:}/{1:}'.format(field, comp)
        CHECK_FOLDER = os.path.isdir(MYDIR)

        # If folder doesn't exist, then create it.

        if not CHECK_FOLDER:
            os.makedirs(MYDIR)
            print("created folder : ", MYDIR)

        else:
            print(MYDIR, "folder already exists.")
        plt.savefig('%s/correlation.pdf'%(MYDIR) , dpi=600)

plt.show()

