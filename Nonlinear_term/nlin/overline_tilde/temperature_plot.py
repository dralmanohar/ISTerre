from pylab import *
from numpy import *
from pyxshells import *
import shtns
# ~ import xsplot
import sys
from derivative_spherical import *
import time

############ defined sh

count = 0
info,r =get_field_info('../../../data/fieldU_9000.E1e4_Ra1.5e4_Pm3_hdiff')
sh = shtns.sht(info['lmax'], info['mmax'], info['mres'])
sh.set_grid(nl_order=2, flags=shtns.sht_gauss)
# ~ sh.print_info()
Y00_1 = sh.sh00_1()
nr = info['nr']
l2 = sh.l*(sh.l+1)
Nlat, Nphi = sh.set_grid(nl_order=2, flags=shtns.sht_gauss)
Nr = (info['ir'][1] - info['ir'][0]) + 1

lmax = info['lmax']
mmax = info['mmax']
mres = info['mres']

time1 = round(info['time'],4)

print ("sh.Nlat =\t", Nlat)
print ("sh.Nphi =\t", Nphi)

############################### file loading

def load_fieds_scalar(A, lcut, mcut):
	count = 0
	info,r =get_field_info(A)
	sh = shtns.sht(info['lmax'], info['mmax'], info['mres'])
	sh.set_grid(nl_order=2, flags=shtns.sht_gauss)
	# ~ sh.print_info()
	Y00_1 = sh.sh00_1()
	nr = info['nr']
	l2 = sh.l*(sh.l+1)
	Nlat, Nphi = sh.set_grid(nl_order=2, flags=shtns.sht_gauss)
	Nr = (info['ir'][1] - info['ir'][0]) + 1
	
	Tr = np.zeros((Nr, Nlat, Nphi))
	
	Ttrr = np.zeros((Nr, Nlat, Nphi))
	
	irs = info['ir'][0]
	ire = info['ir'][1]
	time = info['time']
	
	# ~ print ("loadint time",info['time'])
	
	r = r[irs:ire+1]
	theta = arccos(sh.cos_theta)
	phi = np.linspace(0, 2*pi, Nphi)
	
	try:
		Tlm = load_field(A)
	except FileNotFoundError:
		print('file is missing')
	else:
		for ir in range(Tlm.irs, Tlm.ire+1):
			#print ("count = \t",count)
			count +=1
			t_lm = Tlm.sh(ir).astype(complex128)
						
			## truncated field
			ttr_lm = truncate_l_m(t_lm, lmax, mmax, lcut, mcut)
						
			T = sh.synth(t_lm)
			
			ttrr = sh.synth(ttr_lm)
				
			Tr[ir - Tlm.irs,:,:] = T
			Ttrr[ir - Tlm.irs,:,:] = ttrr
					
	return r, theta, phi, Tr,Ttrr, time


######### computation of derivative d/dr, d/dtheta, and d/dphi
##############derivative of functions###############
##############derivative in sphreical harmonics#####################
##############derivative in sphreical harmonics#####################

def forward_transform(ur, ut, up):
	ur_lm, ut_lm, up_lm = sh.analys(ur, ut, up)
	return ur_lm, ut_lm, up_lm

def inverse_transform(Ur, Ut, Up):
	ur, ut, up = sh.synth(Ur, Ut, Up)
	return ur, ut, up

##### truncate l and m

def truncate_l_m(A, lmax, mmax, lcut, mcut):#, lmax, mmax):
	
	nlm = shtns.nlm_calc(lmax, mmax, mres)
	B = np.zeros(int(nlm), np.dtype('complex128'))
	
	l=0
	start = time.time()

	for i in range(mmax+1):
		for j in range(lmax+1):
			lm = l + j
			if i<=mcut:
				if lm<=lcut:
					idx = sh.idx(lm,i)
					B[idx] = A[idx]
				else:
					break
			else:
				break
		l +=1
	elapsed_time_fl = (time.time() - start) 
	# ~ print ("Time elapse in truncate the field = \t", elapsed_time_fl)
	return B

def save_npy(r, theta, phi, ur, ut, up, br, bt, bp,A=None):
	
	var_dict = {'r':r,'theta': theta, 'phi': phi, 'ur': ur, 'ut': ut, 'up': up, 'br': br, 'bt': bt, 'bp': bp }
	data_dic = {}
	
	for name, idx in var_dict.items():
		data_dic[name] = idx
	
	filename = A 
	np.save((filename +'.npy'), data_dic)
	
	return 0


def save_npy_scalar(r, theta, phi, Tr, Ttrr, A=None):
	
	var_dict = {'r':r,'theta': theta, 'phi': phi, 'Tr': Tr, 'Trr': Ttrr}
	data_dic = {}
	
	for name, idx in var_dict.items():
		data_dic[name] = idx
	
	filename = A 
	np.save((filename +'.npy'), data_dic)
	
	return 0

def load_numpy(filename):
	
	filename = filename + '.npy'
	
	data = np.load(filename, allow_pickle=True)
	
	r     = data.item().get('r')
	theta = data.item().get('theta')
	phi   = data.item().get('phi')
	Ur 	  = data.item().get('ur')
	Ut 	  = data.item().get('ut')
	Up 	  = data.item().get('up')
	Br 	  = data.item().get('br')
	Bt 	  = data.item().get('bt')
	Bp 	  = data.item().get('bp')
	
	return r, theta, phi, Ur, Ut, Up, Br, Bt, Bp
#################### calculations

def snapshot_nonline_computation():
		
		start = time.time()
		
		################ temperature field
		r,theta,phi,T,Ttr, time1 = load_fieds_scalar('fieldT_9000.E1e4_Ra1.5e4_Pm3_hdiff',0, 0)
		
		print (np.amax(T), np.amin(T))
	
		return 0


snapshot_nonline_computation()
