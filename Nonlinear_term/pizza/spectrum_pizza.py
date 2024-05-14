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
import sys
import matplotlib.pylab as plt

sys.path.append('../pizza/pizza-master/python/')

from pizza import *

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


#################################

def my_inverse_transform(A):
	A = A#*(Nt)
	B = np.fft.ifft2(A)#, axes = 1)
	return B

def my_forward_transform(A):
	B = (np.fft.fft2(A, axes = (-1)))#/(Nt))
	return B


def truncate_field(A, mmax, mcut):
	
	B = np.zeros(A.shape, dtype = complex)
	
	for i in range(A.shape[1]):
		B[:mcut, i] = A[:mcut,i]
	return B
	

f = PizzaFields(tag='E1e8Ra5e12C', ivar=1, datadir = '/home/sharmam/Research/ISTerre_Project/Nonlinear_term/data/Pizza/', endian = 'b')

A = np.array([[1,2,3,4],[5,6,7,8]])


r = f.radius

us = f.us
up = f.uphi
T = f.temp
us_m = f.us_m
n_phi_max = f.n_phi_max
mmax = f.m_max

mcut = 1048


us_trunc = truncate_field(us_m, mmax, mcut)

def spectra_2d(A):
	
	B  = np.zeros(A.shape)
	
	for i in range(A.shape[1]):
		for j in range(A.shape[0]):
			if j==0:
				B[j,i] = (A[j,i].real)**2 + (A[j,i].imag)**2
			else:
				B[j,i] = 0.5*((A[j,i].real)**2 + (A[j,i].imag)**2)
	return B
	

spec_full  = spectra_2d(us_m)
spec_trunc = spectra_2d(us_trunc)

x = np.arange(spec_full.shape[0])

fig, axes = plt.subplots(1,1,figsize=(E,F))

axes.loglog(x[1:], spec_full[1:,1000],  'r', lw = lw11)
axes.loglog(x[1:], spec_trunc[1:,1000], 'b', lw = lw11)

axes.set_xlabel(r"$m+1$")
axes.set_ylabel(r"$E$")

fig.tight_layout()

plt.savefig("spectrum_m.pdf",dpi = 1000)

plt.show()
