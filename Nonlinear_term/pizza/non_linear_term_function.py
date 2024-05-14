from pylab import *
from numpy import *
import numba
import sys
import time

############ defined sh

############################### file loading

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
def non_lin_temp(ur, ut, T):
	#Td = np.zeros((self.Nr, self.Nt, self.Np))
	
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

def save_npy(r, theta, ur, ut,temp, durr, durt, dutr, dutt, dTr, dTt, nlinr, nlint, A=None):
	
	var_dict = {'r':r,'theta': theta, 'ur': ur, 'ut': ut, 'temp':temp, 'durr':durr, 'durt': durt, 'dutr': dutr, 'dutt': dutt, 'dTr': dTr, 'dTt': dTt, 'nlinr': nlinr, 'nlint': nlint}
	data_dic = {}
	
	for name, idx in var_dict.items():
		data_dic[name] = idx
	
	filename = A 
	np.save((filename +'.npy'), data_dic)
	
	return 0

#################### calculations

def snapshot_nonline_computation(ur, ut, temp, r, theta, save_path = None, time_stamp=None):
		
		durr, durt, dutr, dutt, dTr, dTt, nlinr, nlint = non_lin_same_vector(ur, ut, temp, r, theta)	
						
		### truncated field 
		save_npy(r, theta, ur[512:1024, 1024:1536], ut[512:1024, 1024:1536], temp[512:1024, 1024:1536], durr[512:1024, 1024:1536], durt[512:1024, 1024:1536], dutr[512:1024, 1024:1536], dutt[512:1024, 1024:1536], dTr[512:1024, 1024:1536], dTt[512:1024, 1024:1536], nlinr[512:1024, 1024:1536], nlint[512:1024, 1024:1536],'{0:}/2d_field{1:}'.format(save_path, time_stamp))

		return 0

