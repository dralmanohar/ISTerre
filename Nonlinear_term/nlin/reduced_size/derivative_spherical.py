import numpy as np
import time

## derivative of functions

def r_deri(A,r):
	B = np.zeros(A.shape)
	start = time.time()
	for i in range(1, len(r) - 1):
		h_i  = r[i] - r[i-1]
		h_ii = r[i+1] - r[i]
		B[i,:,:] = (1/(h_i + h_ii))*((h_i/(h_ii))*(A[i+1,:,:] - A[i,:,:]) + (h_ii/h_i)*(A[i,:,:] - A[i-1,:,:]))	
	elapsed_time_fl = (time.time() - start) 
	print ("Time elapse in r derivative = \t", elapsed_time_fl)
	return B

def theta_deri(A,theta):
	B = np.zeros(A.shape)
	start = time.time()
	for i in range(1, len(theta) - 1):
		h_i  = theta[i] - theta[i-1]
		h_ii = theta[i+1] - theta[i]
		B[:,i,:] = (1/(h_i + h_ii))*((h_i/(h_ii))*(A[:,i+1,:] - A[:,i,:]) + (h_ii/h_i)*(A[:,i,:] - A[:,i-1,:]))
	elapsed_time_fl = (time.time() - start) 
	print ("Time elapse in theta derivative = \t", elapsed_time_fl)
	return B
		
def phi_deri(A,phi):
	B = np.zeros(A.shape)
	start = time.time()
	for i in range(len(phi)):
		if i ==0:
			B[:,:,i] = (A[:,:,i+1] - A[:,:,i])/(phi[i+1] - phi[i])
		elif 0<i<(len(phi)-2):
			B[:,:,i ] = (A[:,:,i+1] - A[:,:,i-1])/(phi[i+1] - phi[i-1])
		else:
			B[:,:,i] = (A[:,:,i] - A[:,:,i-1])/(phi[i] - phi[i-1])
	elapsed_time_fl = (time.time() - start) 
	print ("Time elapse in phi derivative = \t", elapsed_time_fl)
	return B
	
