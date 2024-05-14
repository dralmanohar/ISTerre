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

def load_numpy(filename):
	
	filename = filename
	
	data = np.load(filename, allow_pickle=True)
	
	print (data)
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
	Er = data.item().get('ET')
	Et = data.item().get('ETtr')
	# ~ Ep = data.item().get('Ep')
	# ~ Eut = data.item().get('Eut')
	
	return r, Er, Et

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
			
def plot_merid(r, ct, a , strm=0, czoom=1, rg=0, rm=0, shift=0, title='', levels=20, cbar=1, cmap=cmap, name=None, com=None, phi=None, field=None, name1=None, name2=None):
		"""m = plot_merid()"""
		r = r.reshape(r.size,1)
		ct = ct.reshape(1, ct.size)
		st = sqrt(1-ct*ct)
		x = r*st
		y = r*ct + 1.1*shift
		
		mi, ma = np.amin(a), np.amax(a)
		m = max(-mi, ma)
		print ('max value', m, "\t mi \t= ",mi,"\t ma \t=",ma)
		
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
		MYDIR = ("({0:}_{1:}{2:})/phi_{3:}".format(name, name1, name2, int(np.floor(phi))))
		CHECK_FOLDER = os.path.isdir(MYDIR)
		
		# If folder doesn't exist, then create it.
		if not CHECK_FOLDER:
			os.makedirs(MYDIR)
			print("created folder : ", MYDIR)
		
		else:
			print(MYDIR, "folder already exists.")
		
		# ~ if title !='':
			# ~ if 'r'==com:
				# ~ axes.set_title(r"$({0:}_{1:}{2:})_{3:}(r, \theta, \phi = {4:})$".format(name, name1, name2, com,int(np.floor(phi))), fontsize = 1.1*G)
			# ~ elif 't'==com:
				# ~ com = 'theta'
				# ~ axes.set_title(r"$({0:}_{1:}{2:})_\{3:}(r, \theta, \phi = {4:})$".format(name,name1, name2,'theta',int(np.floor(phi))), fontsize = 1.1*G)
			# ~ elif 'p' == com:
				# ~ com = 'phi'
				# ~ axes.set_title(r"$({0:}_{1:}{2:})_\{3:}(r, \theta, \phi = {4:})$".format(name,name1, name2,com,int(np.floor(phi))), fontsize = 1.1*G)

		# ~ plt.legend(loc = "upper left",  labelspacing=0.0,ncol=1,prop={'size':A})#,handleheight=0.1)

		# ~ plt.tight_layout()	
		plt.savefig("{0:}/nlin_{1:}_{2:}_phi_{3:}_t_9000.pdf".format(MYDIR, name, com, int(np.floor(phi))), dpi = 600)
		plt.savefig("{0:}/nlin_{1:}_{2:}_phi_{3:}_t_9000.png".format(MYDIR,name, com, int(np.floor(phi))), dpi = 600)
		
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
		fig.tight_layout()
		plt.savefig("9000/field_Wp_tp_r_1p00_t_9000.pdf", dpi = 600)
		plt.savefig("9000/field_Wp_tp_r_1p00_t_9000.png", dpi = 600)
	
		
		# ~ if title !='':
			# ~ plt.text(2.6, 1.2, r'%s_%s'%(), fontsize = 20)
		return m


def plot_Spec(filename, B = None, A=None, C = None):
	
	r, Et, Etr = load_numpy_spec(filename)

	Nr, lmax = Et.shape
	# ~ print ("Etot",Euto[1,:])

	# ~ print ("Et.shape",Eut[1,1:])
	
	if B=='l':
		l = np.arange(0,lmax-1)
	elif B=='m':
		l = np.arange(1, lmax)
	
	# ~ if D=='u':
	rlist = [1,Nr//2,Nr-11]
	# ~ elif D=='b':
		# ~ rlist = [4,Nr//2,Nr-11]
	
	field= {'Et':Et, 'Etr':Etr}
		
	# You should change 'test' to your preferred folder.
	MYDIR = ("spec_{0:}".format(A))
	CHECK_FOLDER = os.path.isdir(MYDIR)
	
	# ~ # If folder doesn't exist, then create it.
	if not CHECK_FOLDER:
		os.makedirs(MYDIR)
		print("created folder : ", MYDIR)
	
	else:
		print(MYDIR, "folder already exists.")
	
	for keys, value in field.items():
		name = keys[-1]
		if name=='r':
			name = 'r'
		elif name=='t':
			name = 'theta'
		elif name=='p':
			name = 'phi'
		else:
			name = 'total'
		
		fig, axes = plt.subplots(1,1,figsize=(1.5*E,1.5*F))

		for ir in rlist:
			# ~ print ("values", value[ir, 1:])
			
			if B=='l':
				Ene = value[ir, 1:]
			elif B=='m':
				Ene = value[ir,:-1]
				
				# ~ print ("Ene", Ene)
							
			axes.loglog(l, Ene, label = "r=%g \t "%r[ir])
		
		if B=='l':
			axes.set_xlabel("l")
		elif B=='m':
			axes.set_xlabel("m+1")
		
		if name=='total' or name=='r':
			axes.set_ylabel(r"$(\tilde {F}_{%s})_{%s}$"%(C, name))#, name))
		else:
			axes.set_ylabel(r"$(\tilde {F}_{%s})_{\%s}$"%(C, name))#, name))
			
		grid(which='both')
		
		# ~ if D=='u':
			# ~ axes.set_ylim(1e-1, 1e5)
		# ~ elif D=='b':
		# ~ axes.set_ylim(1e4, 1e9)
		axes.set_xlim(1,100)
		fig.legend(loc='upper right', fontsize = 1.1*G)
		fig.tight_layout()
		# ~ plt.savefig("{0:}/spectrum_{1:}_{2:}_{3:}.pdf".format(MYDIR, A, B, name), dpi=1200)
		plt.savefig("spectrum_{0:}_{1:}_{2:}.pdf".format( A, B, name), dpi=1200)
		plt.savefig("spectrum_{0:}_{1:}_{2:}.png".format( A, B, name), dpi=1200)
		# ~ plt.savefig("{0:}/spectrum_{1:}_{2:}_{3:}.png".format(MYDIR, A, B, name), dpi=1200)
	plt.show()
	return 0

def plot_spec_diff_files(file1, file2, file3, file4, file5, file6, B=None, A=None, C=None):
	
	r, Er1, Et1, Ep1, Euto1 = load_numpy_spec(file1)
	r, Er2, Et2, Ep2, Euto2 = load_numpy_spec(file2)
	r, Er3, Et3, Ep3, Euto3 = load_numpy_spec(file3)
	r, Er4, Et4, Ep4, Euto4 = load_numpy_spec(file4)
	r, Er5, Et5, Ep5, Euto5 = load_numpy_spec(file5)
	r, Er6, Et6, Ep6, Euto6 = load_numpy_spec(file6)
	
	Nr, lmax = Euto1.shape
			
	if B=='l':
		l = np.arange(0,lmax-1)
	elif B=='m':
		l = np.arange(1, lmax)
	
	# ~ field_dict = {'Fuu': Euto1, 'bFuu': Euto2, 'Fubarubar': Etot3, 'bFubarubar': Etot4, 'tFuu':Etot5, 'btFuu': Etot6}
	# ~ field_label = ['uu', ]
	
	rlist = [1,Nr//2,Nr-11]
	
	for ir in rlist:
		fig, axes = plt.subplots(1,1,figsize=(1.1*E,1.1*F))
		
		# You should change 'test' to your preferred folder.
		MYDIR = ("spec_trunc_{0:}/r_{1:}/{2:}_spec".format(A, r[ir], B))
		CHECK_FOLDER = os.path.isdir(MYDIR)
		
		# ~ # If folder doesn't exist, then create it.
		if not CHECK_FOLDER:
			os.makedirs(MYDIR)
			print("created folder : ", MYDIR)
		
		else:
			print(MYDIR, "folder already exists.")
		
		if B=='l':
			E1 = Euto1[ir, 1:]
			E2 = Euto2[ir, 1:]
			E3 = Euto3[ir, 1:]
			E4 = Euto4[ir, 1:]
			E5 = Euto5[ir, 1:]
			E6 = Euto6[ir, 1:]
		elif B=='m':
			E1 = Euto1[ir,:-1]
			E2 = Euto2[ir,:-1]
			E3 = Euto3[ir,:-1]
			E4 = Euto4[ir,:-1]
			E5 = Euto5[ir,:-1]
			E6 = Euto6[ir,:-1]
			
		axes.loglog(l, E1, 'r', lw = 1.5, label = r"$F_{uu}$")
		axes.loglog(l, E2, 'k', lw = 1.5, label = r"$\overline{F_{uu}}$", dashes = (4,2))
		axes.loglog(l, E3, 'b', lw = 1.5, label = r"${F_{\overline{u} \; \overline{u}}}$")
		axes.loglog(l, E4, 'm', lw = 1.5, label = r"$\overline{F_{\overline{u} \; \overline{u}}}$", dashes = (4,2))
		axes.loglog(l, E5, 'g', lw = 1.5, label = r"$\widetilde{F}_{u u}$")
		axes.loglog(l, E6, 'c', lw = 1.5, label = r"$\overline{\widetilde{F}_{u u}}$", dashes = (4,2))
		
		axes.set_title(r"$r = %1.3f$"%(r[ir]))
		
		if B=='l':
			axes.set_xlabel(r"$l$")
		else:
			axes.set_xlabel(r"$m+1$")
		
		axes.set_xlim(1,100)
		axes.set_ylim(1e3,1e14)
		# ~ fig.legend(loc='best', fontsize = 1.1*G, ncol=2)
		if B=='l':
			if ir==rlist[0]:
				# ~ fig.legend(bbox_to_anchor=(0.96,0.9), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=2)
				fig.legend(bbox_to_anchor=(0.99,0.93), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 1.*G, ncol=2)
				# ~ fig.legend(bbox_to_anchor=(0.2,0.27), loc="lower left",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=2)
			else:
				# ~ fig.legend(bbox_to_anchor=(1.0,1.0), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=2)

				fig.legend(bbox_to_anchor=(0.35,0.2), loc="lower left",  bbox_transform=fig.transFigure, fontsize = 1.10*G, ncol=2)
		else:
			if ir==rlist[0]:
				# ~ fig.legend(bbox_to_anchor=(0.96,0.9), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=2)
				fig.legend(bbox_to_anchor=(0.65,0.20), loc="lower right",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=2)
			else:
				# ~ fig.legend(bbox_to_anchor=(0.96,0.9), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=2)
				fig.legend(bbox_to_anchor=(0.2,0.2), loc="lower left",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=2)
		fig.tight_layout()
		
		plt.savefig('%s/kinetic_energy_spectrum_%s_r%1.3f_%s.pdf'%(MYDIR, B, r[ir], C))
		
	return 0



def plot_energy_diff_files(file1, file2, file3, B=None, A=None, C=None):
	
	r, Er1, Et1, Ep1, Euto1 = load_numpy_spec(file1)
	r, Er2, Et2, Ep2, Euto2 = load_numpy_spec(file2)
	r, Er3, Et3, Ep3, Euto3 = load_numpy_spec(file3)
	
	Nr, lmax = Euto1.shape
			
	if B=='l':
		l = np.arange(0,lmax-1)
	elif B=='m':
		l = np.arange(1, lmax)
	
	# ~ field_dict = {'Fuu': Euto1, 'bFuu': Euto2, 'Fubarubar': Etot3, 'bFubarubar': Etot4, 'tFuu':Etot5, 'btFuu': Etot6}
	# ~ field_label = ['uu', ]
	
	rlist = [1,Nr//2,Nr-11]
	
	for ir in rlist:
		fig, axes = plt.subplots(1,1,figsize=(E,F))
		
		# You should change 'test' to your preferred folder.
		MYDIR = ("spec_trunc_{0:}/r_{1:}/{2:}_spec".format(A, r[ir], B))
		CHECK_FOLDER = os.path.isdir(MYDIR)
		
		# ~ # If folder doesn't exist, then create it.
		if not CHECK_FOLDER:
			os.makedirs(MYDIR)
			print("created folder : ", MYDIR)
		
		else:
			print(MYDIR, "folder already exists.")
		
		if B=='l':
			E1 = Euto1[ir, 1:]
			E2 = Euto2[ir, 1:]
			E3 = Euto3[ir, 1:]
			
		elif B=='m':
			E1 = Euto1[ir,:-1]
			E2 = Euto2[ir,:-1]
			E3 = Euto3[ir,:-1]
						
		axes.loglog(l, E1, 'r', lw = 1.5, label = r"$E_{b}$")
		axes.loglog(l, E2, 'k', lw = 1.5, label = r"${E_{\overline{b}}}$", dashes = (4,2))
		axes.loglog(l, E3, 'b', lw = 1.5, label = r"${E_{\widetilde{b}}}$")
		
		axes.set_title(r"$r = %1.3f$"%(r[ir]))
		
		if B=='l':
			axes.set_xlabel(r"$l$")
		else:
			axes.set_xlabel(r"$m+1$")
		
		axes.set_xlim(1,100)
		axes.set_ylim(1e-3,1e6)
		# ~ fig.legend(loc='best', fontsize = 1.1*G, ncol=2)
		if B=='l':
			if ir==rlist[0]:
				# ~ fig.legend(bbox_to_anchor=(0.96,0.9), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=2)
				fig.legend(bbox_to_anchor=(0.25,0.25), loc="lower left",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=2)
			else:
				# ~ fig.legend(bbox_to_anchor=(0.96,0.9), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=2)

				fig.legend(bbox_to_anchor=(0.25,0.25), loc="lower left",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=1)
		else:
			if ir==rlist[0]:
				# ~ fig.legend(bbox_to_anchor=(0.96,0.9), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=2)
				fig.legend(bbox_to_anchor=(1.0,1.0), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=1)
			else:
				# ~ fig.legend(bbox_to_anchor=(0.96,0.9), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=2)
				fig.legend(bbox_to_anchor=(1.0,1.0), loc="upper right",  bbox_transform=fig.transFigure, fontsize = 1.1*G, ncol=1)
		fig.tight_layout()
		
		plt.savefig('%s/kinetic_energy_spectrum_%s_r%1.3f_%s.pdf'%(MYDIR, B, r[ir], C))
		
	return 0


def total_energy(filename):
	
	r, Er, Et, Ep, Eut = load_numpy_spec(filename)
	
	# ~ print ("Etot",Eut[1,:])
	# ~ print ("Etot",E)
	# ~ print ("Etot",Eut[1,:])
	
	# ~ Er, Et, Ep, Eut = Er[1:,], Et[1:,], Ep[1:,], Eut[1:,], 
	
	Nr, lmax = Eut.shape
	
	l = np.arange(0,lmax-1)
	
	print ("l = ",len(l))
	print ("l = ",lmax)
	
	sumr = 0
	sumt = 0
	sump = 0
	sumet = 0
	
	for ir in range(Nr):
		for l in range(lmax):
			sumr += Er[ir, l]
			sumt += Et[ir, l]
			sump += Ep[ir, l]
			sumet += Eut[ir, l]
	
	return sumet #sumr, sumt, sump, sumet
	

## for spectrum plotting

# ~ sumr, sumt, sump, sumet = total_energy('../spectrum_correct/spec_l_full.npy')

file1 = '../../results/9000/trunc_40/final_nonlinear/l_spec/spec_l_fluac_T_snap.npy'
# ~ file2 = '../../results/9000/trunc_40/final_nonlinear/l_spec/spec_l_trunc_full_uu_snap.npy'
# ~ file3 = '../../results/9000/trunc_40/final_nonlinear/l_spec/spec_l_trunc_uu_snap.npy'
# ~ file4 = '../../results/9000/trunc_40/final_nonlinear/l_spec/spec_l_trunc_trunc_uu_snap.npy'
# ~ file5 = '../../results/9000/trunc_40/final_nonlinear/l_spec/spec_l_full_Fuu_snap.npy'
# ~ file6 = '../../results/9000/trunc_40/final_nonlinear/l_spec/spec_l_trunc_full_Fuu_snap.npy'

# ~ plot_spec_diff_files(file1, file2, file3, file4, file5, file6, B='l', A = '40', C='uu')
plot_Spec(file1, B='l', A = '40', C='T')

'''
######## energy
file1 = '../../results/9000/trunc_40/final_nonlinear/l_spec/spec_l_full_b_snap.npy'
file2 = '../../results/9000/trunc_40/final_nonlinear/l_spec/spec_l_trunc_b_snap.npy'
file3 = '../../results/9000/trunc_40/final_nonlinear/l_spec/spec_l_fluac_b_snap.npy'

energyl   = total_energy('../../results/9000/trunc_40/final_nonlinear/m_spec/spec_m_full_b_snap.npy')
energytrl = total_energy('../../results/9000/trunc_40/final_nonlinear/m_spec/spec_m_trunc_b_snap.npy')
energyfl  = total_energy('../../results/9000/trunc_40/final_nonlinear/m_spec/spec_m_fluac_b_snap.npy')

Etot = energytrl + energyfl

f = open('energy_bb_%s.dat'%('l'),'w')
f.write('%f \t %f \t %f \t %f \n'%(energyl, energytrl, energyfl, Etot))
f.close()

print ("energyl =\t",energyl)
print ("Etot = \t",Etot)


plot_energy_diff_files(file1, file2, file3, B='l', A = '40', C='Eb')
'''
plt.show()
#plot_Spec('../9000/m_spec/spec_m_full_uxw.npy', A='spec_uxw_full', B='m', C='{uxw} ', D='b')
# ~ plot_Spec('spec_l_UU_snap.npy', A='UU_full', B='l', C='b ', D='u')

# ~ print (sumet)

