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
from read_write import*


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

def load_numpy(filename):
	
	filename = filename
	
	data = np.load(filename, allow_pickle=True)
	
	# ~ print (data)
	r = data.item().get('r')
	theta = data.item().get('theta')
	phi = data.item().get('phi')
	Ur = data.item().get('ur')
	Ut = data.item().get('ut')
	Up = data.item().get('up')
	Br = data.item().get('br')
	Bt = data.item().get('bt')
	Bp = data.item().get('bp')
	
	return r, theta, phi, Ur, Ut, Up, Br, Bt, Bp

def load_numpy_spec(filename):
	
	filename = filename
	
	data = np.load(filename, allow_pickle=True)
	
	# ~ print (data)
	r = data.item().get('r')
	Er = data.item().get('Er')
	Et = data.item().get('Et')
	Ep = data.item().get('Ep')
	Eut = data.item().get('Eut')
	
	return r, Er, Et, Ep, Eut

# ~ nodisplay = False

# ~ if 'DISPLAY' in os.environ:
	# ~ if len(os.environ['DISPLAY'])>0:
		# ~ nodisplay = False

# ~ if __name__=="__main__":
	# ~ import argparse
    # ~ parser = argparse.ArgumentParser(
        # ~ description='XSPP Python module to load and display xspp output.',
        # ~ formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ~ parser.add_argument('fnames', nargs='+', default=None,
                        # ~ help='list of files to display (one figure per file)')
    # ~ parser.add_argument('--nodisplay', action='store_true',
                        # ~ help='set to inhibit display of the figure (forces a save)')
    # ~ parser.add_argument('-z', '--zoom', type=str, default="1",
                        # ~ help='zoom level for color bar')
    # ~ parser.add_argument('-c', '--components', type=str, default="",
                        # ~ help='component list to display')
    # ~ clin = parser.parse_args()
    # ~ nodisplay = nodisplay + clin.nodisplay

# ~ try:
	# ~ import matplotlib
	# ~ if nodisplay:
		# ~ matplotlib.use("Agg")
	
	# ~ import matplotlib.pyplot as plt
	# ~ import matplotlib.ticker as ticker
	# ~ from matplotlib import cm
	
	# ~ cmap = plt.cm.RdBu_r
	
	# ~ fmt = ticker.ScalarFormatter(useMathText = True)
	# ~ fmt.set_powerlimits((-3,3))

# ~ except:
	# ~ cmap, fmt, plt = None, None, None
	# ~ print("[xsplot] Warning: matplotlib not found, trying gnuplot instead (line plots only).")
    # ~ if __name__ != "__main__":
        # ~ print("[xsplot] Note that xsplot provides a basic matplotlib-like interface: xsplot.plot(), xsplot.show(), ...")

################### functions for plotting

# ~ def load_slice_npy(filename):
	# ~ """load a field slice produced by xspp in numpy, return a dictionary with coordinates and fields"""
	# ~ a = np.load(filename) # load numpy data directly
	
	# ~ if len(a.shape)==2:
		# ~ a = a.reshape(1, a.shape[0], a.shape[1]) # toleranece for 1 component data stored as 2D array
	
	# ~ x = a[0, 1:, 0]
	# ~ y = a[0, 0, 1:]
	# ~ data = a[:, 1:, 1:]
	
	# ~ d = {} ## dictionary
	
	# ~ ## identify the slice type
	# ~ plottype = int(a[0,0,0])>>16
	# ~ if plottype==0: # merid slice
		# ~ d['r'], d['theta'] = x, y
	# ~ elif plottype==1:  # a disc slice (periodic)
		# ~ b = np.empty_like(a)
		# ~ b[:, :, 0:-1] = a[:, :, 1:] # copy
		# ~ b[:, :, -1] = a[:, :, 1] # loop arround
        # ~ b[:,0,-1]   = 2*b[:,0,-2]-b[:,0,-3] # next phi-step (not necessarily a full disc)
		# ~ d['r'], d['phi'] = x, b[0, 0,: ]
		# ~ data = b[:,1:,:]
	# ~ elif plottype==2: ## shperical slice
		# ~ d['theta'], d['phi'] = y, x
		# ~ data = np.transpose(data, (0, 2, 1)) ## exchange axis 1 and 2
	# ~ d['data'] = data
	# ~ name = []
	# ~ for i in range(0, a.shape[0]):
		# ~ # decode field name and component
		# ~ name.append(get_slice_name(int(a[i,0,0])))
	# ~ d['name'] = name
	
	# ~ return d

# ~ def get_slice_name(tag):
    # ~ comp = { 0: '', 1: '_r', 2: r'_\theta', 3: r'_\phi', 4: '_s', 5: '_x', 6: '_y', 7: '_z', 8: '_P', 9: '_T', 15: '_{rms}' }
    # ~ print ("tag =\t",tag)
    # ~ tag = int(tag)
    # ~ print ("tag =\t",tag)
    # ~ if (tag&15) in comp.keys():
        # ~ name = comp[tag&15]
    # ~ else: name = ''
    # ~ if (tag>>4)&4095 > 0:
        # ~ name = chr(((tag >> 4)&63) +64) + name
        # ~ if (tag >> 10)&63 > 0:
            # ~ name = chr(((tag >> 10)&63) +64) + name
    # ~ return name

def get_cmap(name=''):
    cmap = cm.RdBu_r
    if len(name) > 0:
        if   name[0] == 'T':  cmap = cm.inferno   # temperature
        elif name[0] == 'C':  cmap = cm.viridis   # composition
        elif name[0] == 'B':  cmap = cm.PRGn_r    # magnetic field
    return cmap

def get_levels(mi, ma, nlevels, czoom):
	if isinstance(czoom, tuple):
		levels = np.linspacce(czoom[0], czoom[1], nlevels+1)
		ext = 'both'
	else:
		if (mi<0) and (ma>0):
			m = max(-mi, ma)
			mi, ma = -m, m ## symmetric color scale
		if czoom == 1:
			levels = np.linspace(mi, ma, nlevels + 1)
			ext = 'neither'
		else:
			c = (ma + mi)/2
			d = (ma - mi)/2
			levels = np.linspace(c- d/czoom, c + d/czoom, nlevels + 1)
			ext = 'both'
		return levels, ext
			
def plot_merid(r, ct, a , strm=0, czoom=1, rg=0, rm=0, shift=0, title='', levels=20, cbar=1, cmap=cmap, name=None, com=None, phi=None, field=None, name1=None, name2=None, val = None, trunc=None):
		"""m = plot_merid()"""
		r = r.reshape(r.size,1)
		ct = ct.reshape(1, ct.size)
		st = sqrt(1-ct*ct)
		x = r*st
		y = r*ct + 1.1*shift
		
		mi, ma = np.amin(a), np.amax(a)
		m = max(-mi, ma)
		
		print ('max value', m, "\t mi \t= ",mi,"\t ma \t=",ma, "com = ",com)

		
		# ~ if name1=='u':
		# ~ if com=='u':
			# ~ mi, ma = -2e6, 2e6
		# ~ else:
			# ~ mi, ma = -2e6, 2e6
		if name2=='b':
			mi, ma = -2e5, 2e5
		else:
			mi, ma = -2e6, 2e6
		# ~ mi, ma = -2e5, 2e5
		# ~ mi, ma = -2e6, 2e6
		# ~ elif name1=='b':
			# ~ mi, ma = -2e6, 2e6
		# ~ elif name1=='j':
			# ~ mi, ma = -2e7,2e7
		
		fig, axes = plt.subplots(1,1,figsize=(E,F))

		if m>0:
			levels, ext = get_levels(mi, ma, levels, czoom)
			p = axes.contourf(x, y, a, levels, cmap=cmap, extend = ext)
		theta = np.linspace(-pi/2, pi/2, 100)
		if rg>0:
			axes.plot(rg*cos(theta) + 2.1*shift, rg*sin(theta), color = 'gray')
		if rm>0:
			axes.plot(rm*cos(theta) + 1.1*shift, rm*sin(theta), color='gray')
		plt.axis('equal')
		plt.axis('off')
		if m>0 and cbar>0:
			fig.colorbar(p, orientation = 'vertical',pad = 0.1, format = fmt)
		ms = np.amax(abs(strm))
		if ms>0:  # plot contour of strm (stream lines)
			lev2 = np.arrange(ms/18, ms/9)
			axes.contour(np.array(x), np.array(y), strm, lev2, colors='k')
			axes.contour(np.array(x), np.array(y), strm, flipud(-lev2), color='k')
		
		

		# You should change 'test' to your preferred folder.
		MYDIR = ("({0:}_{1:}{2:}_{3:})/phi_{4:}".format(name, name1, name2, trunc, int(np.floor(phi))))
		CHECK_FOLDER = os.path.isdir(MYDIR)
		
		# If folder doesn't exist, then create it.
		if not CHECK_FOLDER:
			os.makedirs(MYDIR)
			print("created folder : ", MYDIR)
		
		else:
			print(MYDIR, "folder already exists.")
		
		if trunc=='trunc_full' or trunc=='trunc_full_zoom':
		
			if title !='':
				if 'r'==com:
					axes.set_title(r"$(\overline{%s}_{%s %s})_{%s}(r, \theta, \phi = %d)$"%(name, name1, name2, com,int(np.floor(phi))), fontsize = 1.1*G)
				elif 't'==com:
					com = 'theta'
					axes.set_title(r"$(\overline{%s}_{%s %s})_{\%s}(r, \theta, \phi = %d)$"%(name, name1, name2,'theta',int(np.floor(phi))), fontsize = 1.1*G)
				elif 'p' == com:
					com = 'phi'
					axes.set_title(r"$(\overline{%s}_{%s %s})_{\%s}(r, \theta, \phi = %d)$"%(name,name1,name2,com,int(np.floor(phi))), fontsize = 1.1*G)
		
		elif trunc=='trunc_trunc' or trunc=='trunc_trunc_zoom':
			
			if title !='':
				if 'r'==com:
					axes.set_title(r"$(\overline{%s}_{\overline{%s} \; \overline{%s}})_{%s}(r, \theta, \phi = %d)$"%(name, name1, name2, com,int(np.floor(phi))), fontsize = 1.1*G)
				elif 't'==com:
					com = 'theta'
					axes.set_title(r"$(\overline{%s}_{\overline{%s} \;  \overline{%s}})_{\%s}(r, \theta, \phi = %d)$"%(name, name1, name2,'theta',int(np.floor(phi))), fontsize = 1.1*G)
				elif 'p' == com:
					com = 'phi'
					axes.set_title(r"$(\overline{%s}_{\overline{%s} \; \overline{%s}})_{\%s}(r, \theta, \phi = %d)$"%(name,name1,name2,com,int(np.floor(phi))), fontsize = 1.1*G)
		elif trunc=='full_F' or trunc=='full_F_zoom':
			
			if title !='':
				if 'r'==com:
					axes.set_title(r"$(\widetilde{%s}_{%s %s})_{%s}(r, \theta, \phi = %d)$"%(name, name1, name2, com,int(np.floor(phi))), fontsize = 1.1*G)
				elif 't'==com:
					com = 'theta'
					axes.set_title(r"$(\widetilde{%s}_{%s %s})_{\%s}(r, \theta, \phi = %d)$"%(name, name1, name2,'theta',int(np.floor(phi))), fontsize = 1.1*G)
				elif 'p' == com:
					com = 'phi'
					axes.set_title(r"$(\widetilde{%s}_{%s %s})_{\%s}(r, \theta, \phi = %d)$"%(name,name1,name2,com,int(np.floor(phi))), fontsize = 1.1*G)
		elif trunc=='trunc_F' or trunc=='trunc_F_zoom':
			if title !='':
				if 'r'==com:
					axes.set_title(r"$(\overline{\widetilde{%s}_{%s %s}})_{%s}(r, \theta, \phi = %d)$"%(name, name1, name2, com,int(np.floor(phi))), fontsize = 1.1*G)
				elif 't'==com:
					com = 'theta'
					axes.set_title(r"$(\overline{\widetilde{%s}_{%s %s}})_{\%s}(r, \theta, \phi = %d)$"%(name, name1, name2,'theta',int(np.floor(phi))), fontsize = 1.1*G)
				elif 'p' == com:
					com = 'phi'
					axes.set_title(r"$(\overline{\widetilde{%s}_{%s %s}})_{\%s}(r, \theta, \phi = %d)$"%(name,name1,name2,com,int(np.floor(phi))), fontsize = 1.1*G)
		
		
			# ~ if com == t:
				# ~ com==theta
				# ~ axes.set_title(r"$(j \times b)_{0:}(r, \theta, \phi = {1:})$".format('\phi',int(np.floor(phi))), fontsize = 1.1*G)
			
			# ~ plt.legend(loc = "upper left",  labelspacing=0.0,ncol=1,prop={'size':A})#,handleheight=0.1)

		fig.tight_layout()	
		# ~ plt.savefig("{0:}/nlin_{1:}_{2:}_phi_{3:}_t_9000.pdf".format(MYDIR, name, com, int(np.floor(phi))), dpi = 600)
		
		if val=='v':
			plt.savefig("{0:}/nlin_{1:}_{2:}_phi_{3:}_{4:}_t_9000.pdf".format(MYDIR, name, com, int(np.floor(phi)), val), dpi = 600)
			plt.savefig("{0:}/nlin_{1:}_{2:}_phi_{3:}_{4:}_t_9000.png".format(MYDIR, name, com, int(np.floor(phi)), val), dpi = 600)
		else:
			val=='d'
			plt.savefig("{0:}/nlin_{1:}_{2:}_phi_{3:}_{4:}_t_9000.pdf".format(MYDIR, name, com, int(np.floor(phi)), val), dpi = 600)
			plt.savefig("{0:}/nlin_{1:}_{2:}_phi_{3:}_{4:}_t_9000.png".format(MYDIR, name, com, int(np.floor(phi)), val), dpi = 600)
		
		return m

		
def plot_merid_field(r, ct, a , strm=0, czoom=1, rg=0, rm=0, shift=0, title='', levels=20, cbar=1, cmap=cmap, name=None, com=None, phi=None, field=None, val = None, trunc=None):
		"""m = plot_merid()"""
		r = r.reshape(r.size,1)
		ct = ct.reshape(1, ct.size)
		st = sqrt(1-ct*ct)
		x = r*st
		y = r*ct + 1.1*shift
		
		mi, ma = np.amin(a), np.amax(a)
		m = max(-mi, ma)
		
		print ('max value', m, "\t mi \t= ",mi,"\t ma \t=",ma, "com = ",com)

		# ~ mi, ma = -5e0, 5e0
		# ~ mi, ma = -5e0, 5e0		
		# ~ if name=='u':
		mi, ma = -1.5e3, 1.5e3
		# ~ elif name1=='b':
			# ~ mi, ma = -2e6, 2e6
		# ~ elif name1=='j':
			# ~ mi, ma = -2e7,2e7
			
		if name=='U':
			name = 'u'
		elif name=='B':
			name='b'
		
		
		fig, axes = plt.subplots(1,1,figsize=(E,F))

		if m>0:
			levels, ext = get_levels(mi, ma, levels, czoom)
			p = axes.contourf(x, y, a, levels, cmap=cmap, extend = ext)
		theta = np.linspace(-pi/2, pi/2, 100)
		if rg>0:
			axes.plot(rg*cos(theta) + 2.1*shift, rg*sin(theta), color = 'gray')
		if rm>0:
			axes.plot(rm*cos(theta) + 1.1*shift, rm*sin(theta), color='gray')
		plt.axis('equal')
		plt.axis('off')
		if m>0 and cbar>0:
			fig.colorbar(p, orientation = 'vertical',pad = 0.1, format = fmt)
		ms = np.amax(abs(strm))
		if ms>0:  # plot contour of strm (stream lines)
			lev2 = np.arrange(ms/18, ms/9)
			axes.contour(np.array(x), np.array(y), strm, lev2, colors='k')
			axes.contour(np.array(x), np.array(y), strm, flipud(-lev2), color='k')
		
		

		# You should change 'test' to your preferred folder.
		MYDIR = ("{0:}_{1:}_{2:}/phi_{3:}".format(name, com, trunc, int(np.floor(phi))))
		CHECK_FOLDER = os.path.isdir(MYDIR)
		
		# If folder doesn't exist, then create it.
		if not CHECK_FOLDER:
			os.makedirs(MYDIR)
			print("created folder : ", MYDIR)
		
		else:
			print(MYDIR, "folder already exists.")
		
		
		if trunc=='full':
			if title !='':
				if 'r'==com:
					axes.set_title(r"${0:}_{1:}(r, \theta, \phi = {2:})$".format(name, com, int(np.floor(phi))), fontsize = 1.1*G)
				elif 't'==com:
					com = 'theta'
					axes.set_title(r"${0:}_\{1:}(r, \theta, \phi = {2:})$".format(name,'theta',int(np.floor(phi))), fontsize = 1.1*G)
				elif 'p' == com:
					com = 'phi'
					axes.set_title(r"$ {0:}_\{1:}(r, \theta, \phi = {2:})$".format(name,com,int(np.floor(phi))), fontsize = 1.1*G)
		elif trunc=='trunc':
			if title !='':
				if 'r'==com:
					axes.set_title(r"$\overline{%s}_%s(r, \theta, \phi = %d)$"%(name, com, int(np.floor(phi))), fontsize = 1.1*G)
				elif 't'==com:
					com = 'theta'
					axes.set_title(r"$\overline{%s}_{\%s}(r, \theta, \phi = %d)$"%(name,'theta',int(np.floor(phi))), fontsize = 1.1*G)
				elif 'p' == com:
					com = 'phi'
					axes.set_title(r"$ \overline{%s}_{\%s}(r, \theta, \phi = %d)$"%(name,com,int(np.floor(phi))), fontsize = 1.1*G)
		elif trunc=='fluac' or trunc=='fluac_zoom' :
			if title !='':
				if 'r'==com:
					axes.set_title(r"$\widetilde{%s}_%s(r, \theta, \phi = %d)$"%(name, com, int(np.floor(phi))), fontsize = 1.1*G)
				elif 't'==com:
					com = 'theta'
					axes.set_title(r"$\widetilde{%s}_{\%s}(r, \theta, \phi = %d)$"%(name,'theta',int(np.floor(phi))), fontsize = 1.1*G)
				elif 'p' == com:
					com = 'phi'
					axes.set_title(r"$ \widetilde{%s}_{\%s}(r, \theta, \phi = %d)$"%(name,com,int(np.floor(phi))), fontsize = 1.1*G)	
		
			# ~ if com == t:
				# ~ com==theta
				# ~ axes.set_title(r"$(j \times b)_{0:}(r, \theta, \phi = {1:})$".format('\phi',int(np.floor(phi))), fontsize = 1.1*G)
			
			# ~ plt.legend(loc = "upper left",  labelspacing=0.0,ncol=1,prop={'size':A})#,handleheight=0.1)

		fig.tight_layout()	
		# ~ plt.savefig("{0:}/nlin_{1:}_{2:}_phi_{3:}_t_9000.pdf".format(MYDIR, name, com, int(np.floor(phi))), dpi = 600)
	
		if val=='v':
			plt.savefig("{0:}/{1:}_{2:}_phi_{3:}_{4:}_t_9000.pdf".format(MYDIR, name, com, int(np.floor(phi)), val), dpi = 600)
			plt.savefig("{0:}/{1:}_{2:}_phi_{3:}_{4:}_t_9000.png".format(MYDIR, name, com, int(np.floor(phi)), val), dpi = 600)
		else:
			val='d'	
			plt.savefig("{0:}/{1:}_{2:}_phi_{3:}_{4:}_t_9000.pdf".format(MYDIR, name, com, int(np.floor(phi)), val), dpi = 600)
			plt.savefig("{0:}/{1:}_{2:}_phi_{3:}_{4:}_t_9000.png".format(MYDIR, name, com, int(np.floor(phi)), val), dpi = 600)
		return m


def plot_disc(r, phi, b,czoom=1, rg=0, rm=0, title='', cmap = cmap, levels = 20):
	r = r.reshape(r.size,1)
	phi = phi.reshape(1,phi.size)
	x = r*cos(phi)
	y = r*sin(phi)
	
	mi, ma = np.amin(b), np.amax(b)
	m = max(-mi, ma)
	print ('max value = \t',m)

	fig, axes = plt.subplots(1,1,figsize=(E,F))

	if m>0.0:
		levels, ext = get_levels(mi, ma, levels, czoom)
		p = axes.contourf(x, y, b, levels, cmap = cmap, extend = ext )
		plt.axis('equal')
		plt.axis('off')
		
		# ~ if czoom==1:
			# ~ clim(-m/czoom, m/czoom)
		
		plt.subplots_adjust(left=0.05, bottom=0.05, right = 0.95, top=0.95, wspace=0.2, hspace=0.2)
		#fmt = ticker.FormatStrFormatter("%.3e")
		fmt = ticker.ScalarFormatter(useMathText=True)
		fmt.set_powerlimits((-3,3))
		fig.colorbar(p, format=fmt)
		
		
		theta = np.linspace(-pi/2, pi/2, 100)
		if rg>0:
			axes.plot(rg*cos(theta), rg*sin(theta), color = 'gray')
			axes.plot(-rg*cos(theta),rg*sin(theta), color = 'gray')
		if rm>0:
			plt.plot(rm*cos(theta), rm*sin(theta), color = 'gray')
			plt.plot(-rm*cos(theta), rm*sin(theta), color ='gray')
		if title !='':
			# ~ plt.figtext(0.65, 0.9, title, fontsize = 28)
			axes.set_title(r"$\omega_{\phi}(r, \theta=30, \phi)$", fontsize = 1.5*G)
		fig.tight_layout()
		plt.savefig("9000/field_Wp_rp_theta_30_t_9000.pdf", dpi = 600)
		plt.savefig("9000/field_Wp_rp_theta_30_t_9000.png", dpi = 600)
	
		return m

def plot_surf(theta, phi, vx, czoom=1, rg_r = 0, title = '', cmap = cmap, levels = 10):
	
	mi, ma = np.amin(vx), np.amax(vx)
	m = max(-mi, ma)
	print ("max value =\t",m)
	if m>0.0:
		t = (90.-theta)*pi/180.
		phi = phi.reshape(phi.size)
		ip = np.mgrid[0:phi.size]
		
		
		ip[ip.size-1] = 0   # ip loops around
		p = (phi[ip] - 180)*pi/360#(phi[ip] - 180)*pi/360  # longitude
		p[ip.size-1] = pi/2         # last phi value.
		
		t = t.reshape(t.size, 1)
		p = p.reshape(1, p.size)
		
		print('=> Aitoff projection')
		al = arccos(cos(t)*cos(p))
		al = al / sin(al)

		x = 2*al * (cos(t)*sin(p))
		y = al * (sin(t)*(p*0+1))
		
		b = vx[:, ip]
		
		fig, axes = plt.subplots(1,1,figsize=(E,F))

		levels, ext = get_levels(mi, ma, levels, czoom)
		p = axes.contourf(x, y, b, levels, cmap = cmap, extend = ext)
		plt.axis('equal')
		plt.axis('off')
		cb=fig.colorbar(p, orientation='horizontal',fraction=0.05,pad=0.01,format=fmt)

		axes.set_xlim(-3.2,3.2)
		axes.set_ylim(-1.6, 1.6)
		
		# ~ plt.plot(2*p, p*0, color = 'gray', linestyle = 'dashed') # show equator
		
		if title !='':
			# ~ plt.figtext(0.65, 0.9, title, fontsize = 28)
			axes.set_title(r"$\omega_{\phi}(r=1.00, \theta, \phi)$", fontsize = 1.5*G)
		# ~ fig.tight_layout()
		plt.savefig("9000/field_Wp_tp_r_1p00_t_9000.pdf", dpi = 600)
		plt.savefig("9000/field_Wp_tp_r_1p00_t_9000.png", dpi = 600)
	
		
		# ~ if title !='':
			# ~ plt.text(2.6, 1.2, r'%s_%s'%(), fontsize = 20)
		return m


def plot_Spec(filename):
	
	r, Er, Et, Ep, Eut = load_numpy_spec(filename)

	Nr, lmax = Eut.shape
		
	l = np.arange(0,lmax-1)
	print (l)
	
	rlist = [1,Nr//2,Nr-11]
	
	fig, axes = plt.subplots(1,1,figsize=(1.5*E,1.5*F))
	
	for ir in rlist:
		axes.loglog(l, Eut[ir,1:], label = "r=%g"%r[ir])

	axes.set_xlabel("m+1")
	axes.set_ylabel("Eu")
	grid(which='both')
	axes.set_ylim(1e-1, 1e5)
	axes.set_xlim(0, 100)
	fig.legend(loc='upper right', fontsize = 1.1*G)
	fig.tight_layout()
	plt.savefig("../results/spectrum_l_m.pdf", dpi=1200)
	plt.savefig("../results/spectrum_l_m.png", dpi=1200)
	plt.show()
	return 0
	
#plot_Spec('spectrum_correct/spec_l_full.npy')

'''
r, theta, phi, Ur, Ut, Up, Wr, Wt, Wp = load_numpy('../../results/9000/trunc_40/field_fluac_snap_b_9179.npy')
# ~ r, theta, phi, _, _, _, _, _, _ = load_numpy('FAA_snap_9179.npy')
cmap = 'jet'

phi_cut = [48]# 0, 48, 96]#, 48, 96]
# ~ phi_arrar = [0, 48, 96]
# ~ phi_cut_d = (180/np.pi)*phi

# ~ field = {'Fuwr':Uwr, 'Fuwt': Uwt, 'Fuwp':Uwp, 'Fjbr':Ur, 'Fjbt': Ut, 'Fjbp':Up, 'Fubr':Wr, 'Fubt':Wt, 'Fubp':Wp}
# ~ field = {'Ur':Ur, 'Ut': Ut, 'Up':Up, 'Wr':Wr, 'Wt': Wt, 'Wp':Wp}#, 'Fubr':Wr, 'Fubt':Wt, 'Fubp':Wp}
field = {'Br':Ur, 'Bt': Ut, 'Bp':Up, 'Jr':Wr, 'Jt': Wt, 'Jp':Wp}#, 'Fubr':Wr, 'Fubt':Wt, 'Fubp':Wp}
# ~ field = {'Fubr':Ur, 'Fubt': Ut, 'Fubp':Up, 'Fbur':Wr, 'Fbut':Wt, 'Fbup':Wp}


## for plot u
factor = 1
for ang in phi_cut:
	degree = phi[ang]*(180/np.pi)
	for keys, value in field.items():
		name = keys[0]
		com =  keys[1]
		# ~ print ("name2", name2)
	
		ur = value[:,:,ang]/factor
		plot_merid_field(r, cos(theta), ur, cmap=cmap, czoom=5, levels=20, title = 'com' , name = name, com = com, phi=degree, field = com, trunc='fluac')
'''

## for plot FUU
# ~ print ("Ur = ",Ur)

cmap = 'jet'
r, theta, phi, _, _, _, _, _, _ = load_numpy('truncated_FAB_sanp_9179.npy')

Udr = read2D("../../results/data/not_mean/train/UB_r_48.h5")
Udt = read2D("../../results/data/not_mean/train/UB_t_48.h5")
Udp = read2D("../../results/data/not_mean/train/UB_p_48.h5")

Uvr = read2D("../../results/data/not_mean/train/BU_r_48.h5")
Uvt = read2D("../../results/data/not_mean/train/BU_t_48.h5")
Uvp = read2D("../../results/data/not_mean/train/BU_p_48.h5")

Udr = Udr
Udt = Udt
Udp = Udp

Uvr = Uvr
Uvt = Uvt
Uvp = Uvp

compo = 'results/data/train/U_r_48.h5'

a = int(compo.split("/")[-1].strip('.h5').split("_")[-1])

# ~ print (a)
# ~ print ("udr =\t",Udr)
# ~ print ("udt =\t",Udt)
# ~ print ("udp =\t",Udp)

# ~ phi_cut = [48]# 0, 48, 96]#, 48, 96]
phi_cut = [a]# 0, 48, 96]#, 48, 96]
# ~ phi_arrar = [0, 48, 96]
# ~ phi_cut_d = (180/np.pi)*phi


# ~ field = {'Urd':Udr, 'Utd': Udt, 'Upd':Udp, 'Urv':Uvr, 'Utv': Uvt, 'Upv':Uvp}

# ~ field = {'Fuurd':Udr, 'Fuutd': Udt, 'Fuupd':Udp, 'Fuurv':Uvr, 'Fuutv': Uvt, 'Fuupv':Uvp }#, 'Fbbr':Wr, 'Fbbt':Wt, 'Fbbp':Wp}
#field = {'Fbbrd':Udr, 'Fbbtd': Udt, 'Fbbpd':Udp, 'Fbbrv':Uvr, 'Fbbtv': Uvt, 'Fbbpv':Uvp }#, 'Fbbr':Wr, 'Fbbt':Wt, 'Fbbp':Wp}
#field = {'Fubrd':Udr, 'Fubtd': Udt, 'Fubpd':Udp, 'Fubrv':Uvr, 'Fubtv': Uvt, 'Fubpv':Uvp }#, 'Fbbr':Wr, 'Fbbt':Wt, 'Fbbp':Wp}
#field = {'Fburd':Udr, 'Fbutd': Udt, 'Fbupd':Udp, 'Fburv':Uvr, 'Fbutv': Uvt, 'Fbupv':Uvp }#, 'Fbbr':Wr, 'Fbbt':Wt, 'Fbbp':Wp}

# ~ factor = 1
# ~ for ang in phi_cut:
	# ~ degree = phi[ang]*(180/np.pi)
	# ~ for keys, value in field.items():
		# ~ name = keys[0]
		# ~ name1 = keys[1]
		# ~ name2 = keys[2]
		# ~ com =  keys[3]
		# ~ val = keys[4]
		# ~ print ("name =\t",name)
		# ~ print ("name1 =\t",name1)
		# ~ print ("name2 =\t",name2)
		# ~ print ("com =\t",com)
		#print ("name2", name2)
	
		# ~ ur = value[:,:]
		# ~ plot_merid(r, cos(theta), ur, cmap=cmap, czoom=5, levels=20, title = 'com' , name = name, com = com, phi=degree, field = com, name1=name1, name2=name2, val = val, trunc='trunc_F')
# ~ plt.show()

# ~ field = {'Fuur':Udr, 'Fuut': Udt, 'Fuup':Udp, 'Fbbr':Uvr, 'Fbbt': Uvt, 'Fbbp':Uvp }#, 'Fbbr':Wr, 'Fbbt':Wt, 'Fbbp':Wp}
field = {'Fubr':Udr, 'Fubt': Udt, 'Fubp':Udp, 'Fbur':Uvr, 'Fbut': Uvt, 'Fbup':Uvp }#, 'Fbbr':Wr, 'Fbbt':Wt, 'Fbbp':Wp}
# ~ field = {'Fbbrd':Udr, 'Fbbtd': Udt, 'Fbbpd':Udp, 'Fbbrv':Uvr, 'Fbbtv': Uvt, 'Fbbpv':Uvp }#, 'Fbbr':Wr, 'Fbbt':Wt, 'Fbbp':Wp}
# ~ field = {'Fubrd':Udr, 'Fubtd': Udt, 'Fubpd':Udp, 'Fubrv':Uvr, 'Fubtv': Uvt, 'Fubpv':Uvp }#, 'Fbbr':Wr, 'Fbbt':Wt, 'Fbbp':Wp}
# ~ field = {'Fburd':Udr, 'Fbutd': Udt, 'Fbupd':Udp, 'Fburv':Uvr, 'Fbutv': Uvt, 'Fbupv':Uvp }#, 'Fbbr':Wr, 'Fbbt':Wt, 'Fbbp':Wp}

factor = 1
for ang in phi_cut:
	degree = phi[ang]*(180/np.pi)
	for keys, value in field.items():
		name = keys[0]
		name1 = keys[1]
		name2 = keys[2]
		com =  keys[3]
		# ~ val = keys[4]
		print ("name =\t",name)
		print ("name1 =\t",name1)
		print ("name2 =\t",name2)
		print ("com =\t",com)
		# ~ #print ("name2", name2)
	
		ur = value[:,:]
		plot_merid(r, cos(theta), ur, cmap=cmap, czoom=5, levels=20, title = 'com' , name = name, com = com, phi=degree, field = com, name1=name1, name2=name2, trunc='trunc_F')
plt.show()



# ~ ### plot for field
#field = {'Ur':Udr, 'Ut': Udt, 'Up':Udp, 'Br':Uvr, 'Bt': Uvt, 'Bp':Uvp} # ~ field = {'Fburd':Udr, 'Fbutd': Udt, 'Fbupd':Udp, 'Fburv':Uvr, 'Fbutv': Uvt, 'Fbupv':Uvp }#, 'Fbbr':Wr, 'Fbbt':Wt, 'Fbbp':Wp}

# ~ factor = 1
# ~ for ang in phi_cut:
	# ~ degree = phi[ang]*(180/np.pi)
	# ~ for keys, value in field.items():
		# ~ name = keys[0]
		# ~ name1 = keys[1]
		# ~ name2 = keys[2]
		# ~ com =  keys[3]
		# ~ val = keys[4]
		# ~ print ("name =\t",name)
		# ~ print ("name1 =\t",name1)
		# ~ print ("name2 =\t",name2)
		# ~ print ("com =\t",com)
		#print ("name2", name2)
	
		# ~ ur = value[:,:]
		# ~ plot_merid(r, cos(theta), ur, cmap=cmap, czoom=5, levels=20, title = 'com' , name = name, com = com, phi=degree, field = com, name1=name1, name2=name2, val = val, trunc='trunc_F')
# ~ plt.show()


#'Upd':Udp, 'Urv':Uvr, 'Utv': Uvt, 'Upv':Uvp}
#field = {'Brd':Udr, 'Btd': Udt, 'Bpd':Udp, 'Brv':Uvr, 'Btv': Uvt, 'Bpv':Uvp}

# ~ factor = 1
# ~ for ang in phi_cut:
	# ~ degree = phi[ang]*(180/np.pi)
	# ~ for keys, value in field.items():
		# ~ name = keys[0]
		# ~ com =  keys[1]
		#val = keys[2]
		# ~ print ("name =\t",name)

		# ~ print ("com =\t",com)
		# ~ #print ("name2", name2)
	
		# ~ ur = value[:,:]	
		# ~ plot_merid_field(r, cos(theta), ur, cmap=cmap, czoom=5, levels=20, title = 'com' , name = name, com = com, phi=degree, field = com, trunc='trunc')
# ~ plt.show()


'''
# ~ plot_merid(r, cos(theta), ur, cmap=cmap, czoom=5, levels=20, title = r'$u_r$')
# ~ plot_disc(r, phi, ur, cmap=cmap, czoom=1, levels=20, title = r'$u_r$')
# ~ plot_surf(theta*180./pi, phi*180./pi, ur, cmap=cmap, czoom=1, levels=10, title = r'$u_r$')

# ~ plt.save(fig)

plt.show()


spectrum
def plot_spec(filename):
	sp = np.load(filename, allow_pickle=True)
	print ("sp = ",sp)
	
	return 0
'''
# ~ r, Er, Et, Ep, Eut = load_numpy_spec('../results/spec_l.npy')

# ~ Nr, lmax = Eut.shape
# ~ print ("Eut[1:,:] = \t", Eut[1,:])

# ~ l = np.arange(0,lmax-1)
# ~ print (l)

# ~ rlist = [1,Nr//2,Nr-11]

# ~ fig, axes = plt.subplots(1,1,figsize=(1.5*E,1.5*F))
#Eut1 = Eut[1,1:]

# ~ for ir in rlist:
#	print ("data in loop ir = ",ir,"\t energy \t",Eut[ir,:])
	# ~ axes.loglog(l, Eut[ir,1:], label = "r=%g"%r[ir])
#axes.loglog(l, Eut1, label = "r=%g"%r[1])

# ~ axes.set_xlabel("m+1")
# ~ axes.set_ylabel("Eu")
# ~ grid(which='both')
# ~ axes.set_ylim(1e-1, 1e5)
# ~ axes.set_xlim(0, 100)
# ~ fig.legend(loc='upper right', fontsize = 1.1*G)
# ~ fig.tight_layout()	

# ~ plt.savefig("../results/Manohar_l_spectrum.pdf")
# ~ plt.show()

'''
# ~ plot_spec('spec_l.npy')



# ~ data = np.load('field_9179.25100.npy', allow_pickle = True)




# ~ plot_vari = {'r', 'theta', 'phi'}
# ~ plot_dict = {'ur', 'ut', 'up', 'br', 'bt', 'bp'}

# ~ r = data.item().get('r')
# ~ theta = data.item().get('theta')
# ~ phi = data.item().get('phi')

# ~ print (data.item().get('bp'))

# ~ if 'r' in plot_vari:
	# ~ if 'theta' in plot_vari:
		# ~ for i in plot_dict:
			# ~ plot_merid(r, cos(theta), data.item().get(i), cmap=cmap, czoom=1, levels=20, title = i)
			
	# ~ if 'phi' in plot_vari:
		# ~ print ("r","phi")
# ~ if 'theta' in plot_vari and 'phi' in plot_vari:
	# ~ print ("theta = ", "phi = \t")


# ~ for i in 
# ~ for i in plot_dict:

# ~ print ("ur =\t",ur)
# ~ print ("ur =\t",ur.shape)
						
# ~ def plot_slice(y, i=0, czoom=1, title_prefix = '', levels = 20, cmap=None, name=None):
	# ~ """plot a field slice produced by xspp in numpy format"""
	# ~ if type(y)==str: # if filename load it
		# ~ y = load_slice_npy(y)
    # ~ # choose colormap:
    # ~ if cmap is None:
        # ~ cmap = get_cmap(y['name'][i])
    # ~ ## chose correct plot type
    # ~ if 'r' in y.keys():
		# ~ if 'theta' in y.keys(): # a merid plot
			# ~ plot_merid(y['r'], cos(y['theta']), y['data'][i, :, :], cmap = cmap, czoom = czoom. levels=levels )
		# ~ elif 'phi' in y.keys(): # a disc plot (periodic)
			# ~ plot_disc(y['r'], y['phi'], y['data'][i,:,:], cmap = cmap, czoom=czoom, levels=levels)
	# ~ elif 'theta' in y.keys() and 'phi' in y.keys():
		# ~ plot_surf(y['theta']*180./np.pi, y['phi']*180./np.pi, y['data'][i, :, :], cmap = cmap, czoom = czoom, levels = levels)
	'''
		
