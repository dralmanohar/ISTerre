import numpy as np
import shtns




def save_npy(r, Er, Et, Ep, Eut, A=None):
	
	var_dict = {'r':r, 'Er': Er, 'Et': Et, 'Ep': Ep, 'Eut': Eut}
	data_dic = {}
	
	for name, idx in var_dict.items():
		data_dic[name] = idx
	
	filename = A 
	np.save((filename +'.npy'), data_dic)
	
	return 0

def load_numpy(filename):
	
	filename = filename
	
	data = np.load(filename, allow_pickle=True)
	
	# ~ print (data) 
	
	r = data.item().get('r')
	theta = data.item().get('theta')
	phi = data.item().get('phi')				## for the current files
	Ur = data.item().get('ur')  				## field  Ur
	Ut = data.item().get('ut')  				## field  Utheta
	Up = data.item().get('up')  				## field  Uphi
	Br = data.item().get('br')  				## Vorticity r
	Bt = data.item().get('bt')  				## Vorticity theta
	Bp = data.item().get('bp')  				## ## Vorticity phi
	
	return r, theta, phi, Ur, Ut, Up, Br, Bt, Bp



def ncal(lmax, mmax):
	return int((mmax+1)*(lmax+1) - mmax*(mmax+1)/2)

def forward_transform(ur, us, ut):
	ur_lm, us_lm, ut_lm = sh.analys(ur, us, ut)
	return ur_lm, us_lm, ut_lm

def inverse_transform(ur_lm, us_lm, ut_lm):
	ur, ut, up = sh.synth(ur_lm, us_lm, ut_lm)
	return ur, ut, up


def cal_spec_l(Ur, Ut, Up):
	# ~ f = open("spec_l_full_b.dat",'w')
	# ~ print ("mmax = \t",mmax)
	Ar = np.zeros((Nr, lmax+2))
	As = np.zeros((Nr, lmax+2))
	At = np.zeros((Nr, lmax+2))
	Et = np.zeros((Nr, lmax+2))
	for ir in range(Nr):
		print ("l = \t",ir)
		ur = Ur[ir,:,:]
		us = Ut[ir,:,:]
		ut = Up[ir,:,:]
		ur_lm, us_lm, ut_lm = forward_transform(ur, us, ut)
		count1 = 0
		l = 0
		for l in range(lmax+1):
			sum_r = 0
			sum_s = 0
			sum_t = 0
			for m in range(mmax+1):   ######## here i corresponds to m and lm corresponds to l
				if m<=l:
					if m==0:
						index   = sh.idx(l,m)
						count1 +=1 
						sum_r += 0.5*((ur_lm[index].real)**2 + (ur_lm[index].imag)**2)
						sum_s += 0.5*(l+1)*(l)*((us_lm[index].real)**2 + (us_lm[index].imag)**2)
						sum_t += 0.5*(l+1)*(l)*((ut_lm[index].real)**2 + (ut_lm[index].imag)**2)
					else:
						index   = sh.idx(l,m)
						count1 +=1 
						sum_r += ((ur_lm[index].real)**2 + (ur_lm[index].imag)**2)
						sum_s += (l+1)*(l)*((us_lm[index].real)**2 + (us_lm[index].imag)**2)
						sum_t += (l+1)*(l)*((ut_lm[index].real)**2 + (ut_lm[index].imag)**2)
				else:
					break
			l += 1
			Ar[ir, l] = sum_r
			As[ir, l] = sum_s
			At[ir, l] = sum_t
			Et[ir, l] = sum_r + sum_s + sum_t
			# ~ f.write('%d \t %5.8f \t %5.8f \t %5.8f \t %5.8f \n'%(l, sum_r, sum_s, sum_t, sum_r + sum_s + sum_t))
		print ("count = \t",count1)
	return Ar, As, At, Et
	
def cal_spec_m(Ur, Ut, Up):
	# ~ f = open("spec_m_full_b.dat",'w')
	# ~ print ("mmax = \t",mmax)
	Ar = np.zeros((Nr, mmax+2))
	As = np.zeros((Nr, mmax+2))
	At = np.zeros((Nr, mmax+2))
	Et = np.zeros((Nr, mmax+2))
	for ir in range(Nr):
		ur = Ur[ir,:,:]
		us = Ut[ir,:,:]
		ut = Up[ir,:,:]
		ur_lm, us_lm, ut_lm = forward_transform(ur, us, ut)
		count1 = 0
		l = 0
		for i in range(mmax+1):
			sum_r = 0
			sum_s = 0
			sum_t = 0
			for j in range(lmax+1):   ######## here i corresponds to m and lm corresponds to l
				lm = l + j
				if lm<=lmax:
					if i==0:
						index   = sh.idx(lm,i)
						count1 +=1 
						sum_r += 0.5*((ur_lm[index].real)**2 + (ur_lm[index].imag)**2)
						sum_s += 0.5*(lm+1)*(lm)*((us_lm[index].real)**2 + (us_lm[index].imag)**2)
						sum_t += 0.5*(lm+1)*(lm)*((ut_lm[index].real)**2 + (ut_lm[index].imag)**2)
					else:
						index   = sh.idx(lm,i)
						count1 +=1 
						sum_r += ((ur_lm[index].real)**2 + (ur_lm[index].imag)**2)
						sum_s += (lm+1)*(lm)*((us_lm[index].real)**2 + (us_lm[index].imag)**2)
						sum_t += (lm+1)*(lm)*((ut_lm[index].real)**2 + (ut_lm[index].imag)**2)
				else:
					break
			l += 1
			Ar[ir, i] = sum_r
			As[ir, i] = sum_s
			At[ir, i] = sum_t
			Et[ir, i] = sum_r + sum_s + sum_t
			# ~ f.write('%d \t %5.8f \t %5.8f \t %5.8f \t %5.8f \n'%(i, sum_r, sum_s, sum_t, sum_r + sum_s + sum_t))
		print ("count = \t",count1)
	return Ar, As, At, Et

def cal_spec_l_1D(Ur, Ut, Up):
	# ~ f = open("spec_l_full_b.dat",'w')
	# ~ print ("mmax = \t",mmax)
	Ar = np.zeros((Nr, lmax+2))
	As = np.zeros((Nr, lmax+2))
	At = np.zeros((Nr, lmax+2))
	Et = np.zeros((Nr, lmax+2))
	for ir in range(Nr):
		print ("l = \t",ir)
		ur = Ur[ir,:]
		us = Ut[ir,:]
		ut = Up[ir,:]
		ur_lm, us_lm, ut_lm = forward_transform(ur, us, ut)
		count1 = 0
		l = 0
		for l in range(lmax+1):
			sum_r = 0
			sum_s = 0
			sum_t = 0
			for m in range(mmax+1):   ######## here i corresponds to m and lm corresponds to l
				if m<=l:
					if m==0:
						index   = sh.idx(l,m)
						count1 +=1 
						sum_r += 0.5*((ur_lm[index].real)**2 + (ur_lm[index].imag)**2)
						sum_s += 0.5*(l+1)*(l)*((us_lm[index].real)**2 + (us_lm[index].imag)**2)
						sum_t += 0.5*(l+1)*(l)*((ut_lm[index].real)**2 + (ut_lm[index].imag)**2)
					else:
						index   = sh.idx(l,m)
						count1 +=1 
						sum_r += ((ur_lm[index].real)**2 + (ur_lm[index].imag)**2)
						sum_s += (l+1)*(l)*((us_lm[index].real)**2 + (us_lm[index].imag)**2)
						sum_t += (l+1)*(l)*((ut_lm[index].real)**2 + (ut_lm[index].imag)**2)
				else:
					break
			l += 1
			Ar[ir, l] = sum_r
			As[ir, l] = sum_s
			At[ir, l] = sum_t
			Et[ir, l] = sum_r + sum_s + sum_t
			# ~ f.write('%d \t %5.8f \t %5.8f \t %5.8f \t %5.8f \n'%(l, sum_r, sum_s, sum_t, sum_r + sum_s + sum_t))
		print ("count = \t",count1)
	return Ar, As, At, Et
	
def cal_spec_m_1D(Ur, Ut, Up):
	# ~ f = open("spec_m_full_b.dat",'w')
	# ~ print ("mmax = \t",mmax)
	Ar = np.zeros((Nr, mmax+2))
	As = np.zeros((Nr, mmax+2))
	At = np.zeros((Nr, mmax+2))
	Et = np.zeros((Nr, mmax+2))
	for ir in range(Nr):
		ur = Ur[ir,:]
		us = Ut[ir,:]
		ut = Up[ir,:]
		ur_lm, us_lm, ut_lm = forward_transform(ur, us, ut)
		count1 = 0
		l = 0
		for i in range(mmax+1):
			sum_r = 0
			sum_s = 0
			sum_t = 0
			for j in range(lmax+1):   ######## here i corresponds to m and lm corresponds to l
				lm = l + j
				if lm<=lmax:
					if i==0:
						index   = sh.idx(lm,i)
						count1 +=1 
						sum_r += 0.5*((ur_lm[index].real)**2 + (ur_lm[index].imag)**2)
						sum_s += 0.5*(lm+1)*(lm)*((us_lm[index].real)**2 + (us_lm[index].imag)**2)
						sum_t += 0.5*(lm+1)*(lm)*((ut_lm[index].real)**2 + (ut_lm[index].imag)**2)
					else:
						index   = sh.idx(lm,i)
						count1 +=1 
						sum_r += ((ur_lm[index].real)**2 + (ur_lm[index].imag)**2)
						sum_s += (lm+1)*(lm)*((us_lm[index].real)**2 + (us_lm[index].imag)**2)
						sum_t += (lm+1)*(lm)*((ut_lm[index].real)**2 + (ut_lm[index].imag)**2)
				else:
					break
			l += 1
			Ar[ir, i] = sum_r
			As[ir, i] = sum_s
			At[ir, i] = sum_t
			Et[ir, i] = sum_r + sum_s + sum_t
			# ~ f.write('%d \t %5.8f \t %5.8f \t %5.8f \t %5.8f \n'%(i, sum_r, sum_s, sum_t, sum_r + sum_s + sum_t))
		print ("count = \t",count1)
	return Ar, As, At, Et

## data

# ~ r, theta, phi, Ur, Ut, Up, Wr, Wt, Wp = load_numpy('../../results/9000/trunc_40/final_nonlinear/truncated_FAB_sanp_9179.npy')
r, theta, phi, Ur, Ut, Up, Wr, Wt, Wp = load_numpy('../../results/9000/trunc_40/field_fluac_snap_b_9179.npy')



lmax = 79
mmax = 63
Nr = len(r)
sh = shtns.sht(lmax, mmax)
Nlat, Nphi = sh.set_grid(nl_order=2, flags=shtns.sht_gauss)

Eurl, Eutl, Eupl, Eul_total = cal_spec_l(Ur, Ut, Up)
Eurm, Eutm, Eupm, Eum_total = cal_spec_m(Ur, Ut, Up)

save_npy(r, Eurl, Eutl, Eupl, Eul_total, "../../results/9000/trunc_40/final_nonlinear/l_spec/spec_l_fluac_b_snap" )

save_npy(r, Eurm, Eutm, Eupm, Eum_total, "../../results/9000/trunc_40/final_nonlinear/m_spec/spec_m_fluac_b_snap" )
