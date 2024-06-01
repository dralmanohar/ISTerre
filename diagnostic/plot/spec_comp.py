import numpy as np
import shtns
import os



def save_npy(Er, Et, Ep, Eut, A=None, path_file = None):
	
	var_dict = {'Er': Er, 'Et': Et, 'Ep': Ep, 'Eut': Eut}
	data_dic = {}
	
	for name, idx in var_dict.items():
		data_dic[name] = idx
	
	filename = A 
	np.save((path_file + '/' + filename +'.npy'), data_dic)
	
	return 0

def load_numpy(filename, path_file = None):
	
	filename = filename
	
	data = np.load(path_file + '/' + filename, allow_pickle=True)
	
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
	Nr = Ur.shape[0]
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
	
	Nr = Ur.shape[0]
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

# ~ r, theta, phi, Ur, Ut, Up, Wr, Wt, Wp = load_numpy('../../results/9000/trunc_40/final_nonlinear/truncated_FAB_sanp_9179.npy')
# ~ r, theta, phi, Ur, Ut, Up, Utrr, Utrt, Utrp = load_numpy('BB_trunc_small_None.npy')


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
number_folder.remove('pred_true_distribution.py')
number_folder.remove('spec_comp.py')

print (number_folder)

path_all_fields = []

for folder in number_folder:

    #print ("folder = \t", folder)
    path_sub = os.listdir(folder)

    Ur = np.load(folder + '/' + 'r_comp' + '/' + "ut_true.npy")
    Ut = np.load(folder + '/' + 't_comp' + '/' + "ut_true.npy")
    Up = np.load(folder + '/' + 'p_comp' + '/' + "ut_true.npy")

    Urp = np.load(folder + '/' + 'r_comp' + '/' + "ut_preds.npy")
    Utp = np.load(folder + '/' + 't_comp' + '/' + "ut_preds.npy")
    Upp = np.load(folder + '/' + 'p_comp' + '/' + "ut_preds.npy")

    lmax = 100
    mmax = 85
    #Nr = len(r)

    sh = shtns.sht(lmax, mmax)
    Nlat, Nphi = sh.set_grid(nl_order=2, flags=shtns.sht_gauss)

    Eurl, Eutl, Eupl, Eul_total = cal_spec_l(Ur, Ut, Up)
    Eurm, Eutm, Eupm, Eum_total = cal_spec_m(Ur, Ut, Up)

    Eurtrl, Euttrl, Euptrl, Eultr_total = cal_spec_l(Urp, Utp, Upp)
    Eurtrm, Euttrm, Euptrm, Eumtr_total = cal_spec_m(Urp, Utp, Upp)

    path_save = path + '/' + folder

    print ("path_save = \t", path_save)

    save_npy(Eurl, Eutl, Eupl, Eul_total, "spec_l_fluac_B_snap", path_file = path_save)
    save_npy(Eurm, Eutm, Eupm, Eum_total, "spec_m_fluac_B_snap", path_file = path_save )

    save_npy(Eurtrl, Euttrl, Euptrl, Eultr_total, "spec_l_fluac_Btr_snap", path_file = path_save )
    save_npy(Eurtrm, Euttrm, Euptrm, Eumtr_total, "spec_m_fluac_Btr_snap", path_file = path_save )



'''
Ur = np.load("ut_true.npy")
Ut = np.load("ut_true.npy")
Up = np.load("ut_true.npy")

Urp = np.load("ut_preds.npy")
Utp = np.load("ut_preds.npy")
Upp = np.load("ut_preds.npy")



lmax = 100
mmax = 85
# ~ Nr = len(r)

sh = shtns.sht(lmax, mmax)
Nlat, Nphi = sh.set_grid(nl_order=2, flags=shtns.sht_gauss)

Eurl, Eutl, Eupl, Eul_total = cal_spec_l(Ur, Ut, Up)
Eurm, Eutm, Eupm, Eum_total = cal_spec_m(Ur, Ut, Up)

Eurtrl, Euttrl, Euptrl, Eultr_total = cal_spec_l(Urp, Utp, Upp)
Eurtrm, Euttrm, Euptrm, Eumtr_total = cal_spec_m(Urp, Utp, Upp)

save_npy(Eurl, Eutl, Eupl, Eul_total, "spec_l_fluac_B_snap" )

save_npy(Eurm, Eutm, Eupm, Eum_total, "spec_m_fluac_B_snap" )

save_npy(Eurtrl, Euttrl, Euptrl, Eultr_total, "spec_l_fluac_Btr_snap" )

save_npy(Eurtrm, Euttrm, Euptrm, Eumtr_total, "spec_m_fluac_Btr_snap" )
'''
