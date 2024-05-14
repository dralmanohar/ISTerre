from pylab import *
from numpy import *
import numba
import sys
import time
sys.path.append('/nfs_scratch/sharmam/Nonlinear_term/pizza/pizza-master/python/')

from pizza import *

###########derivative in real space

@numba.jit(nopython=True)
def r_deri(A,r):
	B = np.zeros(A.shape)
	for i in range(len(r)):
		if i==0:
			B[:,i] = (A[:,i+1] - A[:,i])/(r[i+1] - r[i])
		elif 1<=i<=(len(r)-2):
			h_i  = r[i] - r[i-1]
			h_ii = r[i+1] - r[i]
			B[:,i] = (1/(h_i + h_ii))*((h_i/(h_ii))*(A[:,i+1] - A[:,i]) + (h_ii/h_i)*(A[:,i] - A[:,i-1]))	
		else:
			B[:,i] = (A[:,i] - A[:,i-1])/(r[i] - r[i-1])
			
	return B

@numba.jit(nopython=True)
def theta_deri(A,theta):
	B = np.zeros(A.shape)
	for i in range(len(theta)):
		if i ==0:
			B[i,:] = (A[i+1,:] - A[i,:])/(theta[i+1] - theta[i])
		elif 1<=i<=(len(theta)-2):
			B[i,: ] = (A[i+1,:] - A[i-1,:])/(theta[i+1] - theta[i-1])
		else:
			B[i,:] = (A[i,:] - A[i-1,:])/(theta[i] - theta[i-1])
	return B

############compute the nonlinear term (A.grad)B or (B.grad)A

@numba.jit(nopython=True)
def non_lin_temp(ur, ut, T, r, theta):

    DTr = r_deri(T, r)  			## derivative wrt r for temperature field
    DTt = theta_deri(T, theta)      ## derivative wrt theta

    Td = ur*DTr + (ut/r)*DTt

    return Td

@numba.jit(nopython=True)
def non_lin_same_vector(Ur, Ut, T, r, theta):
	
	Durr = r_deri(Ur, r); 	Durt = theta_deri(Ur, theta)
	
	Dutr = r_deri(Ut, r);	Dutt = theta_deri(Ut, theta)
	
	DTr = r_deri(T,r); DTt = theta_deri(T, theta)
	
	Ar = np.zeros((Ur.shape));	At = np.zeros((Ur.shape))

	for i in range(len(theta)):
		for j in range(len(r)):
			Ar[i, j] = Ur[i, j]*Durr[i, j] + (Ut[i, j]/r[j])*Durt[i, j] - ((Ut[i, j]**2)/(r[j]))
			At[i, j] = Ur[i, j]*Dutr[i, j] + (Ut[i, j]/r[j])*Dutt[i, j] + ((Ur[i, j]* Ut[i, j])/r[j])
	return Durr, Durt, Dutr, Dutt, DTr, DTt, Ar, At

def save_npy(r, theta, ur, ut, temp, durr, durt, dutr, dutt, dTr, dTt, nlinr, nlint, nlintr_r, nlintr_t, fububr, fububt, tfur, tfut, temp_d, ttemp_d, ont, A=None):

    var_dict = {'r':r,'theta': theta, 'ur': ur, 'ut': ut, 'temp':temp, 'durr':durr, 'durt': durt, 'dutr': dutr, 'dutt': dutt, 'dTr': dTr, 'dTt': dTt, 'nlinr': nlinr, 'nlint': nlint, 'nlintr_r':nlintr_r, 'nlintr_t':nlintr_t, 'fububr':fububr, 'fububt':fububt, 'tfur':tfur, 'tfut':tfut, 'temp_d':temp_d, 'ttemp_d':ttemp_d, 'ont':ont}

    data_dic = {}

    for name, idx in var_dict.items():
        data_dic[name] = idx

    filename = A
    np.save((filename +'.npy'), data_dic)

    return 0


def truncate_field(A, mmax, mcut, nphi, n_m_max):
	
	D = spat_spec(A, n_m_max)
	
	B = np.zeros(D.shape, dtype = complex)
	
	for i in range(D.shape[1]):
		B[:mcut, i] = D[:mcut,i]
	
	C = spec_spat(B, nphi)
	
	return C


#################### calculations

def snapshot_nonline_computation(ur, ut, temp, utr_r, utr_t, Ttr, r, theta, mmax=None, mcut=None, nphi=None, n_m_max = None, save_path = None, time_stamp=None):

    durr, durt, dutr, dutt, dTr, dTt, nlinr, nlint = non_lin_same_vector(ur, ut, temp, r, theta)

    durr_tr, durt_tr, dutr_tr, dutt_tr, dTr_tr, dTt_tr, nlinr_tr, nlint_tr = non_lin_same_vector(utr_r, utr_t, Ttr, r, theta)

    Td = non_lin_temp(ur, ut, temp, r, theta)
    Td_tr = non_lin_temp(utr_r, utr_t, Ttr, r, theta)

    F_Td_tr = truncate_field(Td, mmax, mcut, nphi, n_m_max)

    T_Tr_tr = F_Td_tr - Td_tr
    O_T_Tr_tr = truncate_field(T_Tr_tr, mmax, mcut, nphi, n_m_max)

    F_nlinr_tr = truncate_field(nlinr, mmax, mcut, nphi, n_m_max)
    F_nlint_tr = truncate_field(nlint, mmax, mcut, nphi, n_m_max)

    T_nlinr_tr = F_nlinr_tr - nlinr_tr
    T_nlint_tr = F_nlint_tr - nlint_tr

    O_T_nlinr_tr = truncate_field(T_nlinr_tr, mmax, mcut, nphi, n_m_max)
    O_T_nlint_tr = truncate_field(T_nlint_tr, mmax, mcut, nphi, n_m_max)

    print ("nonlinear function = \t", mcut)

    save_npy(r, theta, utr_r, utr_t, Ttr, durr_tr, durt_tr, dutr_tr, dutt_tr, dTr_tr, dTt_tr, T_nlinr_tr, T_nlint_tr, O_T_nlinr_tr, O_T_nlint_tr,nlinr_tr,nlint_tr,F_nlinr_tr, F_nlint_tr, Td, Td_tr, O_T_Tr_tr,'{0:}/tilde_nlin{1:}'.format(save_path, time_stamp))

    return 0

