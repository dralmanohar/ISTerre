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


###############################

cmap = 'jet'
# ~ r, theta, phi, _, _, _, _, _, _ = load_numpy('UB_trunc_0299.npy')

# ~ theta = theta[1:-1]

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

print (number_folder)

path_all_fields = []

for folder in number_folder:

    #print ("folder = \t", folder)
    path_sub = os.listdir(folder)
    #print ("path_sub = \t", path_sub)
    for hist in path_sub:

        print ("folder = \t", folder.split("_")[1], "\t hist = \t", hist.split("_")[0])
            

        path_hist = path + '/' + folder + '/' + hist

        Udr = np.load(path_hist + '/' + "ut_true.npy")
        Uvr = np.load(path_hist + '/' + "ut_preds.npy")

        Udr = Udr.flatten()
        Uvr = Uvr.flatten()

        n1 = len(Udr)
        n2 = len(Uvr)

        interval = max(Udr) - min(Udr)

        bins = interval / np.sqrt(n1)
        print ("interval = \t", interval)
        # ~ bins_array = np.linspace(min(Udr), max(Udr), bins)

        fig, axes = plt.subplots(1,1,figsize=(E,F))

        #rng = [min(min(Udr), min(Uvr)), max(max(Udr), max(Uvr))]

        #deno = rng[1] - rng[0]

        # Creating histogram
        axes.hist(Udr, bins = int(np.sqrt(n1)), color = 'r', histtype=u'step', density=True, label=r'$True$')
        axes.hist(Uvr, bins = int(np.sqrt(n1)), color = 'b', histtype=u'step', density=True, label=r'$Model$')
        axes.set_ylabel(r'$pdf$')
        if hist.split("_")[0]=='r':
            compo = 'r'
        elif hist.split("_")[0]=='t':
            compo = 'theta'
        elif hist.split("_")[0]=='p':
            compo = 'phi'
        if compo=='r':
            axes.set_xlabel(r'$\overline{\tilde{F}}_{%s}^%s$'%(folder.split("_")[1], compo))
        else:
            axes.set_xlabel(r'$\overline{\tilde{F}}_{%s}^\%s$'%(folder.split("_")[1], compo))
        # ~ axes.set_xlim(-700,700)
        # ~ axes.set_ylim(0,4e-3)
        axes.set_yscale('log')
        axes.legend(bbox_to_anchor = (1., 1.), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 0.8*G, ncol=1)
        fig.tight_layout()
        # Show plot
        plt.savefig("%s/histogram.pdf"%(path_hist), dpi = 1200)
plt.show()


        #print ("path_sub = \t", path_hist)

#print ()

'''
Udr = np.load("ut_true.npy")

Uvr = np.load("ut_preds.npy")

Udr = Udr.flatten()
Uvr = Uvr.flatten()


n1 = len(Udr)
n2 = len(Uvr)

interval = max(Udr) - min(Udr)

bins = interval / np.sqrt(n1)

# ~ bins_array = np.linspace(min(Udr), max(Udr), bins)
fig, axes = plt.subplots(1,1,figsize=(E,F))

rng = [min(min(Udr), min(Uvr)), max(max(Udr), max(Uvr))]

deno = rng[1] - rng[0]

# Creating histogram
axes.hist(Udr, bins = int(np.sqrt(n1)), color = 'r', histtype=u'step', density=True, label=r'$True$')
axes.hist(Uvr, bins = int(np.sqrt(n1)), color = 'b', histtype=u'step', density=True, label=r'$Model$')
axes.set_ylabel(r'$pdf$')
axes.set_xlabel(r'$\overline{\tilde{F}}_{uu}^r$')
# ~ axes.set_xlim(-700,700)
# ~ axes.set_ylim(0,4e-3)
axes.set_yscale('log')
axes.legend(bbox_to_anchor = (0.7, 0.6), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 0.8*G, ncol=1)
fig.tight_layout()
# Show plot
plt.savefig("fuu_histogram.pdf", dpi = 1200)
plt.show()
'''
