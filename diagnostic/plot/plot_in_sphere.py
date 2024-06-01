from __future__ import division

import numpy as np
from numpy import sin, cos, arccos, sqrt, pi
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

####################################

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


fmt = ticker.ScalarFormatter(useMathText = True)
fmt.set_powerlimits((-3,3))
################### functions for plotting

cmap = 'jet'

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


##################### plot r and theta
			
def plot_merid(r, ct, a , strm=0, czoom=1, rg=0, rm=0, shift=0, title='', levels=20, cbar=1, cmap=cmap, limit = None,  path_file = None, name=None, com=None, phi=None, field=None, name1=None, name2=None, val = None, trunc=None):
    """m = plot_merid()"""

    r = r.reshape(r.size,1)
    ct = ct.reshape(1, ct.size)
    st = sqrt(1-ct*ct)
    x = r*st
    y = r*ct + 1.1*shift

    print ("shape r = \t", r.shape, "\t ct = \t", ct.shape)

    mi, ma = np.amin(limit), np.amax(limit)

    print ("mi = \t", mi, "\t ma = \t", ma)

    m = max(-mi, ma)

    print ("phi in th eplotting folder = \t", phi)

    #print ('max value', m, "\t mi \t= ",mi,"\t ma \t=",ma, "com = ",com)

    #mi, ma = -0.5e6, 0.5e6

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

    #sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0]))
    #path = sys.path[0]
    #pathsrc = os.path.split(path)[0]#path.split[0]
    #srcpath = os.path.join(pathsrc, 'src')
    #sys.path.append(srcpath)

    # You should change 'test' to your preferred folder.
    MYDIR = path_file + '/' + '{0:}_{1:}{2:}_{3:}/phi_{4:}'.format(name, name1, name2, trunc, int(np.floor(phi)))
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
                axes.set_title(r"$(\overline{%s}_{%s %s})_{%s}(r, \theta, \phi_%d)$"%(name, name1, name2, com,int(np.floor(phi))), fontsize = 1.1*G)

            elif 't'==com:
                com = 'theta'
                axes.set_title(r"$(\overline{%s}_{%s %s})_{\%s}(r, \theta, \phi_%d)$"%(name, name1, name2,'theta',int(np.floor(phi))), fontsize = 1.1*G)

            elif 'p' == com:
                com = 'phi'
                axes.set_title(r"$(\overline{%s}_{%s %s})_{\%s}(r, \theta, \phi_%d)$"%(name,name1,name2,com,int(np.floor(phi))), fontsize = 1.1*G)
    
    elif trunc=='trunc_trunc' or trunc=='trunc_trunc_zoom':

        if title !='':
            if 'r'==com:
                axes.set_title(r"$(\overline{%s}_{\overline{%s} \; \overline{%s}})_{%s}(r, \theta, \phi_%d)$"%(name, name1, name2, com,int(np.floor(phi))), fontsize = 1.1*G)

            elif 't'==com:
                com = 'theta'
                axes.set_title(r"$(\overline{%s}_{\overline{%s} \;  \overline{%s}})_{\%s}(r, \theta, \phi_%d)$"%(name, name1, name2,'theta',int(np.floor(phi))), fontsize = 1.1*G)
            elif 'p' == com:
                com = 'phi'
                axes.set_title(r"$(\overline{%s}_{\overline{%s} \; \overline{%s}})_{\%s}(r, \theta, \phi_%d)$"%(name,name1,name2,com,int(np.floor(phi))), fontsize = 1.1*G)
    
    elif trunc=='full_F' or trunc=='full_F_zoom':

        if title !='':
            if 'r'==com:
                axes.set_title(r"$(\widetilde{%s}_{%s %s})_{%s}(r, \theta, \phi_%d)$"%(name, name1, name2, com,int(np.floor(phi))), fontsize = 1.1*G)

            elif 't'==com:
                com = 'theta'
                axes.set_title(r"$(\widetilde{%s}_{%s %s})_{\%s}(r, \theta, \phi_%d)$"%(name, name1, name2,'theta',int(np.floor(phi))), fontsize = 1.1*G)

            elif 'p' == com:
                com = 'phi'
                axes.set_title(r"$(\widetilde{%s}_{%s %s})_{\%s}(r, \theta, \phi_%d)$"%(name,name1,name2,com,int(np.floor(phi))), fontsize = 1.1*G)
    
    elif trunc=='trunc_F' or trunc=='trunc_F_zoom':

        if title !='':
            if 'r'==com:
                axes.set_title(r"$\overline{\widetilde{%s}_{%s %s}^{%s}}(r, \theta, \phi_%d)$"%(name, name1, name2, com,int(np.floor(phi))), fontsize = 1.1*G)

            elif 't'==com:
                com = 'theta'
                axes.set_title(r"$\overline{\widetilde{%s}_{%s %s}}^{\%s}(r, \theta, \phi_%d)$"%(name, name1, name2,'theta',int(np.floor(phi))), fontsize = 1.1*G)

            elif 'p' == com:
                com = 'phi'
                axes.set_title(r"$\overline{\widetilde{%s}_{%s %s}}^{\%s}(r, \theta, \phi_%d)$"%(name,name1,name2,com,int(np.floor(phi))), fontsize = 1.1*G)

    fig.tight_layout()	

    if val=='v':
        plt.savefig("{0:}/nlin_{1:}_{2:}_phi_{3:}_{4:}_t_9000.pdf".format(MYDIR, name, com, int(np.floor(phi)), val), dpi = 600)
        plt.savefig("{0:}/nlin_{1:}_{2:}_phi_{3:}_{4:}_t_9000.png".format(MYDIR, name, com, int(np.floor(phi)), val), dpi = 600)

    else:
        val=='d'
        plt.savefig("{0:}/nlin_{1:}_{2:}_phi_{3:}_{4:}_t_9000.pdf".format(MYDIR, name, com, int(np.floor(phi)), val), dpi = 600)
        plt.savefig("{0:}/nlin_{1:}_{2:}_phi_{3:}_{4:}_t_9000.png".format(MYDIR, name, com, int(np.floor(phi)), val), dpi = 600)

    return m

############## plot r and phi

def plot_disc(r, phi, b, strm=0, czoom=1, rg=0, rm=0, shift=0, title='', levels=20, cbar=1, cmap=cmap, limit = None,  path_file = None, name=None, com=None, phi1=None, field=None, name1=None, name2=None, val = None, trunc=None):

    r = r.reshape(r.size,1)
    phi = phi.reshape(1,phi.size)
    x = r*cos(phi)
    y = r*sin(phi)

    mi, ma = np.amin(limit), np.amax(limit)
    m = max(-mi, ma)

    print ('max value', m, "\t mi \t= ",mi,"\t ma \t=",ma, "com = ",com)

    #mi, ma = -0.4e6, 0.4e6

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

        # You should change 'test' to your preferred folder.

        MYDIR = path_file + '/' + '{0:}_{1:}{2:}_{3:})/phi_{4:}'.format(name, name1, name2, trunc, int(np.floor(phi1)))
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
                    axes.set_title(r"$(\overline{%s}_{%s %s})_{%s}(r, \theta, \phi_%d)$"%(name, name1, name2, com,int(np.floor(phi1))), fontsize = 1.1*G)

                elif 't'==com:
                    com = 'theta'
                    axes.set_title(r"$(\overline{%s}_{%s %s})_{\%s}(r, \theta, \phi_%d)$"%(name, name1, name2,'theta',int(np.floor(phi1))), fontsize = 1.1*G)

                elif 'p' == com:
                    com = 'phi'
                    axes.set_title(r"$(\overline{%s}_{%s %s})_{\%s}(r, \theta, \phi_%d)$"%(name,name1,name2,com,int(np.floor(phi1))), fontsize = 1.1*G)

        elif trunc=='trunc_trunc' or trunc=='trunc_trunc_zoom':

            if title !='':
                if 'r'==com:
                    axes.set_title(r"$(\overline{%s}_{\overline{%s} \; \overline{%s}})_{%s}(r, \theta, \phi_%d)$"%(name, name1, name2, com,int(np.floor(phi1))), fontsize = 1.1*G)
                elif 't'==com:
                    com = 'theta'
                    axes.set_title(r"$(\overline{%s}_{\overline{%s} \;  \overline{%s}})_{\%s}(r, \theta, \phi_%d)$"%(name, name1, name2,'theta',int(np.floor(phi1))), fontsize = 1.1*G)
                elif 'p' == com:
                    com = 'phi'
                    axes.set_title(r"$(\overline{%s}_{\overline{%s} \; \overline{%s}})_{\%s}(r, \theta, \phi_%d)$"%(name,name1,name2,com,int(np.floor(phi1))), fontsize = 1.1*G)
        elif trunc=='full_F' or trunc=='full_F_zoom':

            if title !='':
                if 'r'==com:
                    axes.set_title(r"$(\widetilde{%s}_{%s %s})_{%s}(r, \theta, \phi_%d)$"%(name, name1, name2, com,int(np.floor(phi1))), fontsize = 1.1*G)

                elif 't'==com:
                    com = 'theta'
                    axes.set_title(r"$(\widetilde{%s}_{%s %s})_{\%s}(r, \theta, \phi_%d)$"%(name, name1, name2,'theta',int(np.floor(phi1))), fontsize = 1.1*G)

                elif 'p' == com:
                    com = 'phi'
                    axes.set_title(r"$(\widetilde{%s}_{%s %s})_{\%s}(r, \theta, \phi_%d)$"%(name,name1,name2,com,int(np.floor(phi1))), fontsize = 1.1*G)

        elif trunc=='trunc_F' or trunc=='trunc_F_zoom':
            if title !='':
                if 'r'==com:
                    axes.set_title(r"$(\overline{\widetilde{%s}_{%s %s}^{%s}})(r, \theta_%d, \phi)$"%(name, name1, name2, com,int(np.floor(phi1))), fontsize = 1.1*G)

                elif 't'==com:
                    com = 'theta'
                    axes.set_title(r"$(\overline{\widetilde{%s}_{%s %s}})_{\%s}(r, \theta, \phi_%d)$"%(name, name1, name2,'theta',int(np.floor(phi1))), fontsize = 1.1*G)

                elif 'p' == com:
                    com = 'phi'
                    axes.set_title(r"$(\overline{\widetilde{%s}_{%s %s}})_{\%s}(r, \theta, \phi_%d)$"%(name,name1,name2,com,int(np.floor(phi1))), fontsize = 1.1*G)

        fig.tight_layout()

        if val=='v':
            plt.savefig("{0:}/nlin_{1:}_{2:}_phi_{3:}_{4:}_t_9000.pdf".format(MYDIR, name, com, int(np.floor(phi1)), val), dpi = 600)
            plt.savefig("{0:}/nlin_{1:}_{2:}_phi_{3:}_{4:}_t_9000.png".format(MYDIR, name, com, int(np.floor(phi1)), val), dpi = 600)

        else:
            val=='d'
            plt.savefig("{0:}/nlin_{1:}_{2:}_phi_{3:}_{4:}_t_9000.pdf".format(MYDIR, name, com, int(np.floor(phi1)), val), dpi = 600)
            plt.savefig("{0:}/nlin_{1:}_{2:}_phi_{3:}_{4:}_t_9000.png".format(MYDIR, name, com, int(np.floor(phi1)), val), dpi = 600)

        return m


########## plot theta and phi
def plot_surf(theta, phi, vx, strm=0, czoom=1, rg=0, rm=0, shift=0, title='', levels=20, cbar=1, cmap=cmap, limit = None, path_file = None, name=None, com=None, phi1=None, field=None, name1=None, name2=None, val = None, trunc=None):
	
	mi, ma = np.amin(limit), np.amax(limit) #-0.3e6, 0.3e6#np.amin(b), np.amax(b)
	m = max(-mi, ma)
	print ("max value =\t",m)
	print ("min value mi =\t",mi)
	print ("min value ma =\t",ma)
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
		cb=fig.colorbar(p, orientation='horizontal', format = fmt)#,fraction=0.05,pad=0.01,format=fmt)

		axes.set_xlim(-3.2,3.2)
		axes.set_ylim(-1.6, 1.6)
		
		# ~ plt.plot(2*p, p*0, color = 'gray', linestyle = 'dashed') # show equator
		
		# You should change 'test' to your preferred folder.
		MYDIR = path_file + '/' + '{0:}_{1:}{2:}_{3:}/phi_{4:}'.format(name, name1, name2, trunc, int(np.floor(phi1)))
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
					axes.set_title(r"$(\overline{%s}_{%s %s})_{%s}(r, \theta, \phi_%d)$"%(name, name1, name2, com,int(np.floor(phi1))), fontsize = 1.1*G)
				elif 't'==com:
					com = 'theta'
					axes.set_title(r"$(\overline{%s}_{%s %s})_{\%s}(r, \theta, \phi_%d)$"%(name, name1, name2,'theta',int(np.floor(phi1))), fontsize = 1.1*G)
				elif 'p' == com:
					com = 'phi'
					axes.set_title(r"$(\overline{%s}_{%s %s})_{\%s}(r, \theta, \phi_%d)$"%(name,name1,name2,com,int(np.floor(phi1))), fontsize = 1.1*G)
		
		elif trunc=='trunc_trunc' or trunc=='trunc_trunc_zoom':
			
			if title !='':
				if 'r'==com:
					axes.set_title(r"$(\overline{%s}_{\overline{%s} \; \overline{%s}})_{%s}(r, \theta, \phi_%d)$"%(name, name1, name2, com,int(np.floor(phi1))), fontsize = 1.1*G)
				elif 't'==com:
					com = 'theta'
					axes.set_title(r"$(\overline{%s}_{\overline{%s} \;  \overline{%s}})_{\%s}(r, \theta, \phi_%d)$"%(name, name1, name2,'theta',int(np.floor(phi1))), fontsize = 1.1*G)
				elif 'p' == com:
					com = 'phi'
					axes.set_title(r"$(\overline{%s}_{\overline{%s} \; \overline{%s}})_{\%s}(r, \theta, \phi_%d)$"%(name,name1,name2,com,int(np.floor(phi1))), fontsize = 1.1*G)
		elif trunc=='full_F' or trunc=='full_F_zoom':
			
			if title !='':
				if 'r'==com:
					axes.set_title(r"$(\widetilde{%s}_{%s %s})_{%s}(r, \theta, \phi_%d)$"%(name, name1, name2, com,int(np.floor(phi1))), fontsize = 1.1*G)
				elif 't'==com:
					com = 'theta'
					axes.set_title(r"$(\widetilde{%s}_{%s %s})_{\%s}(r, \theta, \phi_%d)$"%(name, name1, name2,'theta',int(np.floor(phi1))), fontsize = 1.1*G)
				elif 'p' == com:
					com = 'phi'
					axes.set_title(r"$(\widetilde{%s}_{%s %s})_{\%s}(r, \theta, \phi_%d)$"%(name,name1,name2,com,int(np.floor(phi1))), fontsize = 1.1*G)
		elif trunc=='trunc_F' or trunc=='trunc_F_zoom':
			if title !='':
				if 'r'==com:
					axes.set_title(r"$\overline{\widetilde{%s}^{%s}_{%s %s}}(r_%d, \theta, \phi)$"%(name, com, name1, name2,int(np.floor(phi1))), fontsize = 1.1*G)
				elif 't'==com:
					com = 'theta'
					axes.set_title(r"$\overline{\widetilde{%s}^{\%s}_{%s %s}}(r, \theta, \phi_%d)$"%(name, com, name1, name2, int(np.floor(phi1))), fontsize = 1.1*G)
				elif 'p' == com:
					com = 'phi'
					axes.set_title(r"$\overline{\widetilde{%s}^{\%s}_{%s %s}}(r, \theta, \phi_%d)$"%(name,com, name1,name2,int(np.floor(phi1))), fontsize = 1.1*G)
		

		fig.tight_layout()	

		if val=='v':
			plt.savefig("{0:}/nlin_{1:}_{2:}_phi_{3:}_{4:}_t_9000.pdf".format(MYDIR, name, com, int(np.floor(phi1)), val), dpi = 600)
			plt.savefig("{0:}/nlin_{1:}_{2:}_phi_{3:}_{4:}_t_9000.png".format(MYDIR, name, com, int(np.floor(phi1)), val), dpi = 600)
		else:
			val=='d'
			plt.savefig("{0:}/nlin_{1:}_{2:}_phi_{3:}_{4:}_t_9000.pdf".format(MYDIR, name, com, int(np.floor(phi1)), val), dpi = 600)
			plt.savefig("{0:}/nlin_{1:}_{2:}_phi_{3:}_{4:}_t_9000.png".format(MYDIR, name, com, int(np.floor(phi1)), val), dpi = 600)
		
	
		return m

	
