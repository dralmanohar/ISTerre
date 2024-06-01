from __future__ import division

import numpy as np
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
import os

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


cmap = plt.cm.RdBu_r

fmt = ticker.ScalarFormatter(useMathText = True)
fmt.set_powerlimits((-3,3))



def load_numpy_spec(filename, path_file=None):

    filename = filename
    print ("path_file = \t", path_file)

    data = np.load(path_file + '/' + filename, allow_pickle=True)

    # ~ print (data)
    Er = data.item().get('Er')
    Et = data.item().get('Et')
    Ep = data.item().get('Ep')
    Eut = data.item().get('Eut')

    return Er, Et, Ep, Eut

def plot_energy_diff_files(file1, file2, B=None, A=None, C=None, comp = None, path_file = None):

    _,_,_,Eutob1 = load_numpy_spec(file1, path_file = path_file)
    _,_,_,Eutob2 = load_numpy_spec(file2, path_file = path_file)

    Nr, lmax = Eutob1.shape

    if B=='l':
        l = np.arange(0,lmax-1)

    elif B=='m':
        l = np.arange(1, lmax)

    rlist = [Nr//2]

    for ir in rlist:
        fig, axes = plt.subplots(1,1,figsize=(E,F))

        # You should change 'test' to your preferred folder.
        MYDIR = ("{0:}/{1:}_spec".format(path_file, B))
        CHECK_FOLDER = os.path.isdir(MYDIR)

        # ~ # If folder doesn't exist, then create it.

        if not CHECK_FOLDER:
            os.makedirs(MYDIR)
            print("created folder : ", MYDIR)

        else:
            print(MYDIR, "folder already exists.")

        if B=='l':
            Eb1 = Eutob1[ir, 1:]
            Eb2 = Eutob2[ir, 1:]

        elif B=='m':
            Eb1 = Eutob1[ir,:-1]
            Eb2 = Eutob2[ir,:-1]

        print ("Eb1 = \t", Eb1)
        axes.loglog(l, Eb1, 'r', lw = 1.5, label = r"${{\overline{\tilde{F}_{%s}}}}^{True}$"%(comp))
        axes.loglog(l, Eb2, 'k', lw = 1.5, label = r"${{\overline{\tilde{F}_{%s}}}}^{pred}$"%(comp), dashes = (4,2))

        if B=='l':
            axes.set_xlabel(r"$l$")
            axes.set_ylabel(r"$E$")
        else:
            axes.set_xlabel(r"$m+1$")
            axes.set_ylabel(r"$E$")

        axes.set_xlim(1,50)
        axes.set_ylim(1e8,1e11)
        # ~ fig.legend(loc='best', fontsize = 1.1*G, ncol=2)

        if B=='l':
            if ir==rlist[0]:
                # ~ fig.legend(bbox_to_anchor=(0.96,0.9), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=2)
                fig.legend(bbox_to_anchor=(0.25,0.6), loc="lower left",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=1)

            else:
                # ~ fig.legend(bbox_to_anchor=(0.96,0.9), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=2)
                fig.legend(bbox_to_anchor=(0.25,0.6), loc="lower left",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=1)

        else:
            if ir==rlist[0]:
                # ~ fig.legend(bbox_to_anchor=(0.96,0.9), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=2)
                fig.legend(bbox_to_anchor=(0.9,0.9), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 1.*G, ncol=2)
            else:
                # ~ fig.legend(bbox_to_anchor=(0.96,0.9), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=2)
                fig.legend(bbox_to_anchor=(1.0,0.60), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=1)
                fig.tight_layout()
        plt.savefig('%s/magnetic_energy_spectrum_%s_r_%s.pdf'%(MYDIR, B, C))

    return 0

path = './'
number_folder = os.listdir(path)

print (number_folder)


number_folder.remove('To_get_full_field_stack.py')
number_folder.remove('outlog')
number_folder.remove('0_32')
number_folder.remove('32_64')
#number_folder.remove('f_bu')
#number_folder.remove('f_bb')
number_folder.remove('histogram_plot.py')
number_folder.remove('pred_true_distribution.py')
number_folder.remove('spec_comp.py')
number_folder.remove('plot_spectrum.py')

print (number_folder)

path_all_fields = []

for folder in number_folder:

    #print ("folder = \t", folder)
    path_sub = os.listdir(folder)
    print ("path = \t", path_sub)
    
    path_data = folder 
    #print ("compo = \t", folder.split("_")[1])
    file1 = 'spec_m_fluac_B_snap.npy'
    file2 = 'spec_m_fluac_Btr_snap.npy'
    plot_energy_diff_files(file1, file2, B='m', A = '50', C='Eu', comp = folder.split("_")[1], path_file = path_data)

plt.show()

