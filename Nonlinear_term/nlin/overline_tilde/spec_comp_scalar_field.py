
import numpy as np
import shtns




def save_npy(r, ET, ETtr, A=None):
	
	var_dict = {'r':r, 'ET': ET, 'ETtr':ETtr}
	data_dic = {}
	
	for name, idx in var_dict.items():
		data_dic[name] = idx
	
	filename = A 
	np.save((filename +'.npy'), data_dic)
	
	return 0

def load_numpy(filename):
	
	filename = filename
	
	data = np.load(filename, allow_pickle=True)
	
	# ~ print ("data =\t",data)
	# ~ print (data) 
	
	r = data.item().get('r')
	theta = data.item().get('theta')
	phi = data.item().get('phi')				## for the current files
	Tr = data.item().get('Tr')  				## field  Ur
	Ttr = data.item().get('Trr')  				## field  Utheta
	## ## Vorticity phi
	
	return r, theta, phi, Tr, Ttr


def ncal(lmax, mmax):
	return int((mmax+1)*(lmax+1) - mmax*(mmax+1)/2)

def forward_transform(ur):
	ur_lm = sh.analys(ur)
	return ur_lm

def inverse_transform(ur_lm):
	ur = sh.synth(ur_lm)
	return ur


def cal_spec_l(Tr, Ttrr):
	# ~ f = open("spec_l_full_b.dat",'w')
	# ~ print ("mmax = \t",mmax)
	ET = np.zeros((Nr, lmax+2))
	ETtr = np.zeros((Nr, lmax+2))

	for ir in range(Nr):
		print ("l = \t",ir)
		T = Tr[ir,:,:]
		Ttr = Ttrr[ir,:,:]
	
		Tr_lm    = forward_transform(T)
		Ttr_lm  = forward_transform(Ttr)
		
		count1 = 0
		
		l = 0
		for l in range(lmax+1):
			sum_T = 0
			sum_Ttr = 0
		
			for m in range(mmax+1):   ######## here i corresponds to m and lm corresponds to l
				if m<=l:
					if m==0:
						index   = sh.idx(l,m)
						count1 +=1 
						sum_T   += 0.5*((Tr_lm[index].real)**2  + (Tr_lm[index].imag)**2)
						sum_Ttr += 0.5*((Ttr_lm[index].real)**2 + (Ttr_lm[index].imag)**2)
						
					else:
						index   = sh.idx(l,m)
						count1 +=1 
						sum_T   += ((Tr_lm[index].real)**2  + (Tr_lm[index].imag)**2)
						sum_Ttr += ((Ttr_lm[index].real)**2 + (Ttr_lm[index].imag)**2)
						
				else:
					break
			l += 1
			ET[ir, l] = sum_T
			ETtr[ir, l] = sum_Ttr
			# ~ f.write('%d \t %5.8f \t %5.8f \t %5.8f \t %5.8f \n'%(l, sum_r, sum_s, sum_t, sum_r + sum_s + sum_t))
		print ("count = \t",count1)
	return ET, ETtr
	
def cal_spec_m(Tr, Ttrr):
	# ~ f = open("spec_m_full_b.dat",'w')
	ET = np.zeros((Nr, lmax+2))
	ETtr = np.zeros((Nr, lmax+2))
	
	
	for ir in range(Nr):
		print ("l = \t",ir)
		T = Tr[ir,:,:]
		Ttr = Ttrr[ir,:,:]
	
		Tr_lm    = forward_transform(T)
		Ttr_lm  = forward_transform(Ttr)
		
		count1 = 0
		
		l = 0
		for i in range(mmax+1):
			sum_T = 0
			sum_Ttr = 0
			for j in range(lmax+1):   ######## here i corresponds to m and lm corresponds to l
				lm = l + j
				if lm<=lmax:
					if i==0:
						index   = sh.idx(lm,i)
						count1 +=1 
						sum_T   += 0.5*((Tr_lm[index].real)**2  + (Tr_lm[index].imag)**2)
						sum_Ttr += 0.5*((Ttr_lm[index].real)**2 + (Ttr_lm[index].imag)**2)
					else:
						index   = sh.idx(lm,i)
						count1 +=1 
						sum_T   += ((Tr_lm[index].real)**2  + (Tr_lm[index].imag)**2)
						sum_Ttr += ((Ttr_lm[index].real)**2 + (Ttr_lm[index].imag)**2)
				else:
					break
			l += 1
			ET[ir, l] = sum_T
			ETtr[ir, l] = sum_Ttr
			# ~ f.write('%d \t %5.8f \t %5.8f \t %5.8f \t %5.8f \n'%(i, sum_r, sum_s, sum_t, sum_r + sum_s + sum_t))
		print ("count = \t",count1)
	return ET, ETtr

## data

# ~ r, theta, phi, Ur, Ut, Up, Wr, Wt, Wp = load_numpy('../../results/9000/trunc_40/final_nonlinear/truncated_FAB_sanp_9179.npy')
r, theta, phi, Tr, Ttr = load_numpy('../../results/9000/trunc_40/field_T_trunc_snap_9179.npy')

# ~ print ("Ttr",Ttr)


lmax = 79
mmax = 63
Nr = len(r)
sh = shtns.sht(lmax, mmax)
Nlat, Nphi = sh.set_grid(nl_order=2, flags=shtns.sht_gauss)

ETl, ETrl = cal_spec_l(Tr, Ttr)
ETm, ETrm = cal_spec_m(Tr, Ttr)

save_npy(r, ETl, ETrl, "../../results/9000/trunc_40/final_nonlinear/l_spec/spec_l_fluac_T_snap" )

save_npy(r, ETm, ETrm, "../../results/9000/trunc_40/final_nonlinear/m_spec/spec_m_fluac_T_snap" )
