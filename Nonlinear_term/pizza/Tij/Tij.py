from pylab import *
from numpy import *
import numba
import sys
import time

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
def non_lin_temp(ur, ut, tp, T):
	#Td = np.zeros((self.Nr, self.Nt, self.Np))
	
	DTr = r_deri(T, r)  			## derivative wrt r for temperature field
	DTt = theta_deri(T, theta)      ## derivative wrt theta
	DTp = phi_deri(T, phi)        	## derivative wrt phi
	
	sint = np.sin(theta)
	
	Td = ur*DTr + (ut/r)*DTt + (up/r*sint)*DTp
	
	return Td

@numba.jit(nopython=True)
def derivative_vector_scalar(Ur, Ut, T, r, theta):
	
	Durr = r_deri(Ur, r); 	Durt = theta_deri(Ur, theta)
	
	Dutr = r_deri(Ut, r);	Dutt = theta_deri(Ut, theta)
	
	DTr = r_deri(T,r); DTt = theta_deri(T, theta)
	
	Ar = np.zeros((Ur.shape));	At = np.zeros((Ur.shape))

	return Durr, Durt, Dutr, Dutt, DTr, DTt

def save_npy(r, theta, ur, ut, temp, durr, durt, dutr, dutt, dTr, dTt, Trr, Trt, Ttt, Trr_tr, Trt_tr, Ttt_tr, A=None):
	
	var_dict = {'r':r,'theta': theta, 'ur': ur, 'ut': ut, 'temp':temp, 'durr':durr, 'durt': durt, 'dutr': dutr, 'dutt': dutt, 'dTr': dTr, 'dTt': dTt, 'Trr': Trr, 'Trt': Trt, 'Ttt':Ttt, 'Trr_tr':Trr_tr, 'Trt_tr':Trt_tr, 'Ttt_tr':Ttt_tr}
	data_dic = {}
	
	for name, idx in var_dict.items():
		data_dic[name] = idx
	
	filename = A 
	np.save((filename +'.npy'), data_dic)
	
	return 0


def truncate_field(A, mmax, mcut):
	
	B = np.zeros(A.shape, dtype = complex)
	
	for i in range(A.shape[1]):
		B[:mcut, i] = A[:mcut,i]
	return B


#################### calculations

def snapshot_nonline_computation(ur, ut, temp, utr_r, utr_t, Ttr, r, theta, mmax=None, mcut=None,save_path = None, time_stamp=None):
	
	uurr = truncate_field(ur*ur, mmax, mcut)
	uurt = truncate_field(ur*ut, mmax, mcut)
	uutt = truncate_field(ut*ut, mmax, mcut)
	
	uurr_tr = utr_r* utr_r
	uurt_tr = utr_r* utr_t
	uutt_tr = utr_t* utr_t
	
	Tuurr = uurr - uurr_tr 
	Tuurt = uurt - uurt_tr 
	Tuutt = uutt - uutt_tr 
	
	O_Tuurr = truncate_field(Tuurr, mmax, mcut)
	O_Tuurt = truncate_field(Tuurt, mmax, mcut)
	O_Tuutt = truncate_field(Tuutt, mmax, mcut)
	
	
	Durr, Durt, Dutr, Dutt, DTr, DTt = derivative_vector_scalar(utr_r, utr_t, temp_tr, r, theta)
	
	save_npy(r, theta, utr_r[512:1024, 1024:1536], utr_t[512:1024, 1024:1536], Ttr[512:1024, 1024:1536], Durr[512:1024, 1024:1536], Durt[512:1024, 1024:1536], Dutr[512:1024, 1024:1536], Dutt[512:1024, 1024:1536], DTr[512:1024, 1024:1536], DTt[512:1024, 1024:1536], Tuurr[512:1024, 1024:1536], Tuurt[512:1024, 1024:1536], Tuutt[512:1024, 1024:1536], O_Tuurr[512:1024, 1024:1536], O_Tuurt[512:1024, 1024:1536], O_Tuutt[512:1024, 1024:1536],'{0:}/Tij_{1:}'.format(save_path, time_stamp))
	
	return 0

