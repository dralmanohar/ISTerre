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
from plot_in_sphere import*
matplotlib.rcParams.update(matplotlib.rcParamsDefault)


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

	r = data.item().get('r')
	Er = data.item().get('Er')
	Et = data.item().get('Et')
	Ep = data.item().get('Ep')
	Eut = data.item().get('Eut')
	
	return r, Er, Et, Ep, Eut
	
####################################



cmap = 'jet'
r, theta, phi, _, _, _, _, _, _ = load_numpy('UB_trunc_0299.npy')

path = './'
number_folder = os.listdir(path)

#print (number_folder)


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
number_folder.remove('plot_in_sphere.py')
number_folder.remove('UB_trunc_0299.npy')
number_folder.remove('plot_nonliner_full_sphere_real_space.py')
number_folder.remove('__pycache__')

print (number_folder)

path_all_fields = []


for folder in number_folder:
    path_sub = os.listdir(folder)
    path_sub.remove('m_spec')
    path_sub.remove('spec_m_fluac_Btr_snap.npy')
    path_sub.remove('spec_l_fluac_Btr_snap.npy')
    path_sub.remove('spec_m_fluac_B_snap.npy')
    path_sub.remove('spec_l_fluac_B_snap.npy')
    
    for sub_dir in path_sub:

        path_data = path +  folder + '/' + sub_dir
#        print ("sub_dir = \t", path_data)

        
        Udr = np.load(path_data + '/' + "ut_true.npy")
        Uvr = np.load(path_data + '/' + "ut_preds.npy")

        Udr = Udr[10,:,:]
        Uvr = Uvr[10,:,:]

        dUd = Udr - Uvr

        print ("shape = Udr = \t",Udr.shape)
        print ("shape = UUr = \t",Uvr.shape)

        phi_cut = [48]
        print ("shape  = \t", theta.shape)

        folder_name = folder.split('_')[1]
        field_comp = sub_dir.split('_')[0]

        field = {'F%s%sd'%(folder_name, field_comp):Udr, 'F%s%sv'%(folder_name, field_comp):Uvr}
        factor = 1
        #field = {'Fuudv':Udr, 'Fuurv':Uvr, 'Fuufr':dUd}
        # ~ field = {'Fuudv':Uvr}
        factor = 1

        for ang in phi_cut:
            degree = 0#phi[ang]*(180/np.pi)

        for keys, value in field.items():
            name = keys[0]
            name1 = keys[1]
            name2 = keys[2]
            com =  keys[3]
            val = keys[4]

            print ("val = \t", val)

            if val=='d':
                limit = value[:, :]
                print ("Manohar")
            ur = value 
            plot_surf(theta*180/(np.pi), (phi)*180/(np.pi), ur, cmap=cmap, czoom=5, levels=20, title = 'com', limit = limit,  path_file = path_data, name = name, com = com, phi1=degree, field = com, name1=name1, name2=name2, val = val, trunc='trunc_F')

            #plot_surf(theta*180/(np.pi), (phi)*180/(np.pi), ur, cmap=cmap, czoom=5, levels=20, title = 'com' ,  name = name, com = com, phi1=degree, field = com, name1=name1, name2=name2, val = val, trunc='trunc_F')
		# ~ plot_merid(r, cos(theta), ur, cmap=cmap, czoom=5, levels=20, title = 'com' , name = name, com = com, phi=degree, field = com, name1=name1, name2=name2, val = val, trunc='trunc_F')
plt.show()






'''
path = './'

number_folder = os.listdir(path)


number_folder.remove('evaluate_ML.py')
number_folder.remove('path.txt')
number_folder.remove('plot_in_sphere.py')
number_folder.remove('plot_nonliner_real_space.py')
number_folder.remove('read_write.py')
number_folder.remove('UB_trunc_0299.npy')
number_folder.remove('__pycache__')


print ("number_folder = \t", type(number_folder))
print ("number_folder = \t", (number_folder))

constant_dir = 't'

constant_value = 10

for folder in number_folder:

    path_folder = path + '/' + folder
    
    number_subdir = os.listdir(path_folder)

    print ("folder = \t", folder, "\t number sudir = \t", number_subdir)

    for subdir in number_subdir:
        path_sub  = path_folder + '/' + subdir + '/' + 'notebooks'
        path_file = path_folder + '/' + subdir + '/' + 'notebooks' + '/' + 'model_unet'
        print ("path_sub = \t", path_sub, "\t path_file = \t", path_file)
        
        Udr = np.load(path_file + '/' + "label_train.npy")

        Uvr = np.load(path_file + '/' + "preds_train.npy")

        mean_output = np.load(path_file + '/' + "mean_output.npy")
        std_output  = np.load(path_file + '/' + "std_output.npy")

        print ("mean_output = \t",mean_output)
        print ("std_output = \t",std_output)

        Udr = Udr[0,0]

        Udr = Udr*std_output + mean_output

        Uvr = Uvr[0,0]
        Uvr = Uvr*std_output + mean_output

        if constant_dir=='r':    ############# for constant r
            Udr = Udr[constant_value,:,:]
            Uvr = Uvr[constant_value,:,:]
        elif constant_dir=='t':     ################ for constant theta
            Udr = Udr[:, constant_value,:]
            Uvr = Uvr[:, constant_value,:]
        elif constant_dir=='p':   ############### constant phi
            Udr = Udr[:, :, constant_value]
            Uvr = Uvr[:, :, constant_value]

            
        print ("shape = Udr = \t",Udr.shape)

        phi_cut = [48]

        radius = r[50-16:50+16]

        folder_name = path_folder.split('_')[1]
        field_comp = subdir.split('_')[0]

        print ("folder_name = \t", folder_name, "\t filed_comp = \t", field_comp)
        
        print ("path folder = \t", path_sub)

        field = {'F%s%sd'%(folder_name, field_comp):Udr, 'F%s%sv'%(folder_name, field_comp):Uvr}
        factor = 1


        for ang in phi_cut:
            degree = 0#phi[ang]*(180/np.pi)
        limit = 0
        print ("path file before plot = \t", field)
        for keys, value in field.items():
            name = keys[0]
            name1 = keys[1]
            name2 = keys[2]
            com =  keys[3]
            val = keys[4]

            if val=='d':
                limit = value[:, :]
                print ("limit in the loop = \t", limit)
             #   print ("in the directory loop name = \t",name)
                #print ("in the directory loop name1 = \t",name1)
                #print ("in the directory loop name2 = \t",name2)
                #print ("in the directory loop comp = \t",com)
                #print ("in the directory loop value = \t",val)
            ur = value[:,:] #) + mean_output

            print ("valus in the loop = \t", ur)
            print ("shape of r in the loop = \t", radius.shape)
            #plot_surf(theta[0:32]*180/(np.pi), (phi)*180/(np.pi), ur, cmap=cmap, czoom=5, levels=20, title = 'com' , limit = limit,  path_file = path_sub, name = name, com = com, phi1=degree, field = com, name1=name1, name2=name2, val = val, trunc='trunc_F')

            #plot_merid(radius , cos(theta[76-16:76+16]), ur, cmap=cmap, czoom=5, levels=20, title = 'com', limit = limit, path_file = path_sub , name = name, com = com, phi=degree, field = com, name1=name1, name2=name2, val = val, trunc='trunc_F')

            plot_disc(radius, phi[0:64], ur[:,0:64], cmap=cmap, czoom=5, levels=20, title = 'com' , limit = limit, path_file = path_sub, name = name, com = com, phi1=degree, field = com, name1=name1, name2=name2, val = val, trunc='trunc_F')

		# ~ plot_merid(r, cos(theta), ur, cmap=cmap, czoom=5, levels=20, title = 'com' , name = name, com = com, phi=degree, field = com, name1=name1, name2=name2, val = val, trunc='trunc_F')

plt.show()



# ~ theta = theta[1:-1]

Udr = np.load("model_unet/label_train.npy")

Uvr = np.load("model_unet/preds_train.npy")

mean_output = np.load("model_unet/mean_output.npy")
std_output  = np.load("model_unet/std_output.npy")

print ("mean_output = \t",mean_output)
print ("std_output = \t",std_output)

Udr = Udr[0,0]

# ~ print ("max = \t", Udr)

Udr = Udr*std_output + mean_output

Uvr = Uvr[0,0]
print ("max = \t", amax(Uvr))
Uvr = Uvr*std_output + mean_output

Udr = Udr[10,:,:]
Uvr = Uvr[10,:,:]

print ("shape = Udr = \t",Udr.shape)

phi_cut = [48]

r = r[50-16:50+16]

field = {'Fuurd':Udr, 'Fuurv':Uvr}
factor = 1
for ang in phi_cut:
	degree = 0#phi[ang]*(180/np.pi)
	for keys, value in field.items():
		name = keys[0]
		name1 = keys[1]
		name2 = keys[2]
		com =  keys[3]
		val = keys[4]
			
		ur = value[:,:] #) + mean_output
		# ~ plot_merid(r , cos(theta[76-16:76+16]), ur, cmap=cmap, czoom=5, levels=20, title = 'com' , name = name, com = com, phi=degree, field = com, name1=name1, name2=name2, val = val, trunc='trunc_F')
		# ~ plot_disc(r, phi[0:64], ur[:,0:64], cmap=cmap, czoom=5, levels=20, title = 'com' , name = name, com = com, phi1=degree, field = com, name1=name1, name2=name2, val = val, trunc='trunc_F')
		plot_surf(theta[128:152]*180/(np.pi), (phi)*180/(np.pi), ur, cmap=cmap, czoom=5, levels=20, title = 'com' , name = name, com = com, phi1=degree, field = com, name1=name1, name2=name2, val = val, trunc='trunc_F')
		# ~ plot_merid(r, cos(theta), ur, cmap=cmap, czoom=5, levels=20, title = 'com' , name = name, com = com, phi=degree, field = com, name1=name1, name2=name2, val = val, trunc='trunc_F')
plt.show()
'''
