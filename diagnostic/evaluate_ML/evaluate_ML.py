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
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
#plt.style.use(['no-latex'])
#import latex 
import subprocess
from subprocess import call

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

cmap = 'jet'

# ~ theta = theta[1:-1]

path = './'

number_folder = os.listdir(path)

number_folder.remove('evaluate_ML.py')
number_folder.remove('path.txt')
number_folder.remove('plot_in_sphere.py')
number_folder.remove('plot_nonliner_real_space.py')
#number_folder.remove('__pycache__')
number_folder.remove('read_write.py')
number_folder.remove('UB_trunc_0299.npy')
#number_folder.remove('f_ub')
#number_folder.remove('f_bu')
#number_folder.remove('f_bb')

print ("number_folder = \t", type(number_folder))


for folder in number_folder:

    path_folder = path + '/' + folder
    
    number_subdir = os.listdir(path_folder)

    print ("folder = \t", folder, "\t number sudir = \t", number_subdir)

    for subdir in number_subdir:
        path_sub = path_folder + '/' + subdir + '/' + 'notebooks'
        
        print ("paht_sub  = \t", path_sub)

        #subprocess.run(['python', path_sub + '/' + 'eval_unet_loop.py' ])
        call(['python', path_sub + '/' + 'eval_unet_loop.py' ])
        
        



