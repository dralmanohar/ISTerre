from pylab import *
from numpy import *
from pyxshells import *
import shtns
import numba
from numba import set_num_threads
# ~ import xsplot
import sys
from derivative_spherical import *
import time


#set_num_threads(int(16))

############ defined sh

count = 0

############################### file loading

def environment_sht(lmax, mmax, mres, order=None):

    sh = shtns.sht(lmax, mmax, mres)
    sh.set_grid(nl_order = order, flags=shtns.sht_gauss)
    Y00_1 = sh.sh00_1()
    l2 = sh.l*(sh.l+1)
    Nlat, Nphi = sh.set_grid(nl_order=2, flags=shtns.sht_gauss)
    theta = arccos(sh.cos_theta)
    phi   = np.linspace(0, 2*pi, Nphi)

    return sh, theta, phi, Nlat, Nphi


def load_fieds_scalar(A, lcut, mcut):

    count = 0
    info,r =get_field_info(A)
    sh = shtns.sht(info['lmax'], info['mmax'], info['mres'])
    sh.set_grid(nl_order=2, flags=shtns.sht_gauss)
    Y00_1 = sh.sh00_1()
    nr = info['nr']
    l2 = sh.l*(sh.l+1)
    Nlat, Nphi = sh.set_grid(nl_order=2, flags=shtns.sht_gauss)
    Nr = (info['ir'][1] - info['ir'][0]) + 1

    lmax = info['lmax']
    mmax = info['mmax']
    mres = info['mres']

    sh1, theta_tr, phi_tr, Nlat_tr, Nphi_tr = environment_sht(lcut, mcut, mres, order = 2)

    Tr = np.zeros((Nr, Nlat, Nphi))
    Ttrr = np.zeros((Nr, Nlat_tr, Nphi_tr))

    irs = info['ir'][0]
    ire = info['ir'][1]
    time = info['time']

    r = r[irs:ire+1]
    theta = arccos(sh.cos_theta)
    phi = np.linspace(0, 2*pi, Nphi)

    try:
        Tlm = load_field(A)
    except FileNotFoundError:
        print('file is missing')
    else:
        for ir in range(Tlm.irs, Tlm.ire+1):

            count +=1
            t_lm = Tlm.sh(ir).astype(complex128)

            ttrr = truncate_l_m_s(t_lm, lmax, mmax, lcut, mcut, mres, sh, sh1)

            T = sh.synth(t_lm)

            Tr[ir - Tlm.irs,:,:] = T
            Ttrr[ir - Tlm.irs,:,:] = ttrr

    return lmax, mmax, mres, r, theta, phi, theta_tr, phi_tr, Tr,Ttrr, time

def load_fieds(A, lcut, mcut):
    count = 0
    info,r =get_field_info(A)
    sh = shtns.sht(info['lmax'], info['mmax'], info['mres'])
    sh.set_grid(nl_order=2, flags=shtns.sht_gauss)
    Y00_1 = sh.sh00_1()
    nr = info['nr']
    l2 = sh.l*(sh.l+1)
    Nlat, Nphi = sh.set_grid(nl_order=2, flags=shtns.sht_gauss)
    Nr = (info['ir'][1] - info['ir'][0]) + 1

    lmax = info['lmax']
    mmax = info['mmax']
    mres = info['mres']

    sh1, theta_tr, phi_tr, Nlat_tr, Nphi_tr = environment_sht(lcut, mcut, mres, order = 2)

    Ur = np.zeros((Nr, Nlat, Nphi)); 	Ut = np.zeros((Nr, Nlat, Nphi));	Up = np.zeros((Nr, Nlat, Nphi))
    Utrr = np.zeros((Nr, Nlat_tr, Nphi_tr));	Utrt = np.zeros((Nr, Nlat_tr, Nphi_tr));	Utrp = np.zeros((Nr, Nlat_tr, Nphi_tr))

    irs = info['ir'][0]
    ire = info['ir'][1]
    time = info['time']

    r = r[irs:ire+1]
    theta = arccos(sh.cos_theta)
    phi = np.linspace(0, 2*pi, Nphi)

    try:
        Ulm = load_field(A)
    except FileNotFoundError:
        print ('file is missing')
    else:
        print ('loading the file')

        for ir in range(Ulm.irs, Ulm.ire+1):

            Ulm.curl = 0

            count +=1

            ur_lm = Ulm.rad(ir).astype(complex128)
            us_lm = Ulm.sph(ir).astype(complex128)
            ut_lm = Ulm.tor(ir).astype(complex128)

            utrr, utrt, utrp  = truncate_l_m(ur_lm, us_lm, ut_lm, lmax, mmax,  lcut, mcut, mres, sh, sh1)

            ur, ut, up = sh.synth(ur_lm, us_lm, ut_lm)
            
            Ur[ir - Ulm.irs,:,:] = ur
            Ut[ir - Ulm.irs,:,:] = ut
            Up[ir - Ulm.irs,:,:] = up

            Utrr[ir - Ulm.irs,:,:] = utrr
            Utrt[ir - Ulm.irs,:,:] = utrt
            Utrp[ir - Ulm.irs,:,:] = utrp
    return irs, ire, r, theta, phi, theta_tr, phi_tr, Ur, Ut, Up, Utrr, Utrt, Utrp

	
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

@numba.jit(nopython=True)

def r_deri(A,r):
    B = np.zeros(A.shape)
    for i in range(len(r)):
        if i==0:
            B[i,:,:] = (A[i+1,:,:] - A[i,:,:])/(r[i+1] - r[i])
        elif 1<=i<=(len(r)-2):
            h_i  = r[i] - r[i-1]
            h_ii = r[i+1] - r[i]
            B[i,:,:] = (1/(h_i + h_ii))*((h_i/(h_ii))*(A[i+1,:,:] - A[i,:,:]) + (h_ii/h_i)*(A[i,:,:] - A[i-1,:,:]))
        else:
            B[i,:,:] = (A[i,:,:] - A[i-1,:,:])/(r[i] - r[i-1])
    return B

@numba.jit(nopython=True)	
def theta_deri(A,theta):
    B = np.zeros(A.shape)
    for i in range(len(theta)):
        if i==0:
            B[:,i,:] = (A[:,i+1,:] - A[:,i,:])/(theta[i+1] - theta[i])
        elif 1<=i<=(len(theta)-2):
            h_i  = theta[i] - theta[i-1]
            h_ii = theta[i+1] - theta[i]
            B[:,i,:] = (1/(h_i + h_ii))*((h_i/(h_ii))*(A[:,i+1,:] - A[:,i,:]) + (h_ii/h_i)*(A[:,i,:] - A[:,i-1,:]))
        else:
            B[:,i,:] = (A[:,i,:] - A[:,i-1,:])/(theta[i] - theta[i-1])
    return B

@numba.jit(nopython=True)
def phi_deri(A,phi):
    B = np.zeros(A.shape)
    for i in range(len(phi)):
        if i ==0:
            B[:,:,i] = (A[:,:,i+1] - A[:,:,i])/(phi[i+1] - phi[i])
        elif 1<=i<=(len(phi)-2):
            B[:,:,i ] = (A[:,:,i+1] - A[:,:,i-1])/(phi[i+1] - phi[i-1])
        else:
            B[:,:,i] = (A[:,:,i] - A[:,:,i-1])/(phi[i] - phi[i-1])
    return B

############compute the nonlinear term (A.grad)B or (B.grad)A

@numba.jit(nopython=True)
def non_lin_temp(ur, ut, up, T, r, theta, phi):

    DTr = r_deri(T, r)
    DTt = theta_deri(T, theta)
    DTp = phi_deri(T, phi)
    sint = np.sin(theta)

    Td = np.zeros((T.shape))

    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            for k in range(T.shape[2]):
                Td[i,j,k] = ur[i,j,k]*DTr[i,j,k] + (ut[i,j,k]/r[i])*DTt[i,j,k] + (up[i,j,k]/(r[i]*sint[j]))*DTp[i, j, k]

    return Td

@numba.jit(nopython=True)
def non_lin_diff_vector(Ar, At, Ap, Br, Bt, Bp, r, theta, phi, AgB=None):

    if 'uGb'==AgB:
        Ur, Ut, Up, Br, Bt, Bp = Ar, At, Ap, Br, Bt, Bp
    elif 'bGu'==AgB:
        Br, Bt, Bp, Ur, Ut, Up = Ar, At, Ap, Br, Bt, Bp

    DBrr = r_deri(Br, r);			DBtr = r_deri(Bt, r);			DBpr = r_deri(Bp, r)
    DBrt = theta_deri(Br, theta);	DBtt = theta_deri(Bt, theta);	DBpt = theta_deri(Bp, theta)
    DBrp = phi_deri(Br, phi);		DBtp = phi_deri(Bt, phi);		DBpp = phi_deri(Bp, phi)
    ABr = np.zeros(Ar.shape);		ABt = np.zeros(Ar.shape);		ABp = np.zeros(Ar.shape)

    for i in range(Ar.shape[0]):
        for j in range(Ar.shape[1]):
            for k in range(Ar.shape[2]):
                ABr[i, j, k] = Ur[i, j, k]*DBrr[i, j, k] + (Ut[i, j, k]/r[i])* DBrt[i, j, k] + (Up[i, j, k]/(r[i]* sin(theta[j])))* DBrp[i, j, k]  - ((Ut[i, j, k]*Bt[i, j, k])/r[i]) - ((Up[i, j, k]*Bp[i, j, k])/r[i])
                ABt[i, j, k] = Ur[i, j, k]*DBtr[i, j, k] + (Ut[i, j, k]/r[i])* DBtt[i, j, k] + (Up[i, j, k]/(r[i]* sin(theta[j])))* DBtp[i, j, k]  + ((Ut[i, j, k]*Br[i, j, k])/r[i]) - ((Up[i, j, k]*Bp[i, j, k]* cos(theta[j]))/(r[i]* sin(theta[j])))
                ABp[i, j, k] = Ur[i, j, k]*DBpr[i, j, k] + (Ut[i, j, k]/r[i])* DBpt[i, j, k] + (Up[i, j, k]/(r[i]* sin(theta[j])))* DBpp[i, j, k]  + ((Up[i, j, k]*Br[i, j, k])/r[i]) + ((Up[i, j, k]*Bt[i, j, k]* cos(theta[j]))/(r[i]* sin(theta[j])))
    return ABr, ABt, ABp

@numba.jit(nopython=True)
def non_lin_same_vector(Ur, Ut, Up, r, theta, phi):

    Durr = r_deri(Ur, r); 	Durt = theta_deri(Ur, theta);	Durp = phi_deri(Ur, phi)
    Dutr = r_deri(Ut, r);	Dutt = theta_deri(Ut, theta);	Dutp = phi_deri(Ut, phi)
    Dupr = r_deri(Up, r);	Dupt = theta_deri(Up, theta);	Dupp = phi_deri(Up, phi)
    Ar = np.zeros((Ur.shape));	At = np.zeros((Ur.shape));	Ap = np.zeros((Ur.shape))

    for i in range(len(r)):
        for j in range(len(theta)):
            for k in range(len(phi)):
                Ar[i, j, k] = Ur[i, j, k]*Durr[i, j, k] + (Ut[i, j, k]/r[i])*Durt[i, j, k] + (Up[i, j, k]/(r[i]*sin(theta[j])))*Durp[i, j, k] - ((Ut[i, j, k]**2 + Up[i, j, k]**2)/(r[i]))
                At[i, j, k] = Ur[i, j, k]*Dutr[i, j, k] + (Ut[i, j, k]/r[i])*Dutt[i, j, k] + (Up[i, j, k]/(r[i]*sin(theta[j])))*Dutp[i, j, k] + ((Ur[i, j, k]* Ut[i, j, k])/r[i]) - (cos(theta[j])/(r[i]*sin(theta[j])))*(Up[i, j, k]**2)
                Ap[i, j, k] = Ur[i, j, k]*Dupr[i, j, k] + (Ut[i, j, k]/r[i])*Dupt[i, j, k] + (Up[i, j, k]/(r[i]*sin(theta[j])))*Dupp[i, j, k] + ((Ur[i, j, k]* Up[i, j, k])/r[i]) + (cos(theta[j])/(r[i]*sin(theta[j])))*(Ut[i, j, k]*Up[i, j, k])
    return Ar, At, Ap

##### truncate l and m


def truncate_l_m_s(Ar, lmax, mmax, lcut, mcut, mres, sh1, sh2):#, lmax, mmax):

    nlm = shtns.nlm_calc(lcut, mcut, mres)
    Atrr = np.zeros(int(nlm), np.dtype('complex128'))

    l=0

    for i in range(mmax+1):
        for j in range(lmax+1):
            lm = l + j
            if i<=mcut:
                if lm<=lcut:
                    idx1 = sh1.idx(lm,i)   ##
                    idx2 = sh2.idx(lm,i)   ##
                    Atrr[idx2] = Ar[idx1]
                else:
                    break
            else:
                break
        l +=1

    Ur  = sh2.synth(Atrr)

    return Ur


def truncate_l_m(Ar, At, Ap, lmax, mmax, lcut, mcut, mres, sh1, sh2):#, lmax, mmax):

    nlm = shtns.nlm_calc(lcut, mcut, mres)

    Atrr = np.zeros(int(nlm), np.dtype('complex128'))
    Atrt = np.zeros(int(nlm), np.dtype('complex128'))
    Atrp = np.zeros(int(nlm), np.dtype('complex128'))

    l=0

    for i in range(mmax+1):
        for j in range(lmax+1):
            lm = l + j
            if i<=mcut:
                if lm<=lcut:
                    idx1 = sh1.idx(lm,i)   ##
                    idx2 = sh2.idx(lm,i)   ##

                    Atrr[idx2] = Ar[idx1]
                    Atrt[idx2] = At[idx1]
                    Atrp[idx2] = Ap[idx1]
                else:
                    break
            else:
                break
        l +=1

    Ur, Ut, Up = sh2.synth(Atrr, Atrt, Atrp)

    return Ur, Ut, Up



def save_npy(r, theta, phi, ur, ut, up, br, bt, bp,A=None):

    var_dict = {'r':r,'theta': theta, 'phi': phi, 'ur': ur, 'ut': ut, 'up': up, 'br': br, 'bt': bt, 'bp': bp }
    data_dic = {}

    for name, idx in var_dict.items():
        data_dic[name] = idx

    filename = A
    np.save((filename +'.npy'), data_dic)

    return 0


def save_npy_scalar(r, theta, phi, Tr, Ttrr, deri, nlin, nlintr, A=None):
    var_dict = {'r':r,'theta': theta, 'phi': phi, 'Tr': Tr, 'Trr': Ttrr, 'deri':deri, 'nlin':nlin, 'nlintr':nlintr}
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

def grad_vector(Vr, Vt, Vp, r, theta, phi):

    DVrr = r_deri(Vr, r);  DVrt = theta_deri(Vr, theta); DVrp = phi_deri(Vr, phi)
    DVtr = r_deri(Vt, r);  DVtt = theta_deri(Vt, theta); DVtp = phi_deri(Vt, phi)
    DVpr = r_deri(Vp, r);  DVpt = theta_deri(Vp, theta); DVpp = phi_deri(Vp, phi)

    GT11 = DVrr
    GT12 = np.zeros((Vr.shape))
    GT13 = np.zeros((Vr.shape))
    GT21 = DVtr
    GT22 = np.zeros((Vr.shape))
    GT23 = np.zeros((Vr.shape))
    GT31 = DVpr
    GT32 = np.zeros((Vr.shape))
    GT33 = np.zeros((Vr.shape))

    for i in range(len(r)):
        for j in range(len(theta)):
            for k in range(len(phi)):
                GT12[i,j,k] = (1/r[i])*DVrt[i,j,k] - (Vt[i,j,k]/r[i])
                GT13[i,j,k] = (1/(r[i]*np.sin(theta[j])))*DVrp[i, j, k] - (Vp[i, j, k]/r[i])
                GT22[i,j,k] = (1/r[i])*DVtt[i, j, k] + (Vr[i, j, k]/r[i])
                GT23[i,j,k] = (1/(r[i]*np.sin(theta[j])))* (DVtp[i, j, k] - Vp[i, j, k]*np.cos(theta[j]))
                GT32[i,j,k] = (1/r[i])*DVpt[i, j, k]
                GT33[i,j,k] = (1/(r[i]*np.sin(theta[j])))*(DVpp[i, j, k] + Vr[i, j, k]*np.sin(theta[j]) + Vt[i, j, k]*np.cos(theta[j]))
    return	GT11, GT12, GT13, GT21, GT22, GT23, GT31, GT32, GT33, DVrr, DVrt, DVrp, DVtr, DVtt, DVtp, DVpr, DVpt, DVpp

@numba.jit(nopython=True)
def grad_scalar(T, r, theta, phi):

    DTr = r_deri(T, r)
    DTt = theta_deri(T, theta)
    DTp = phi_deri(T, phi)
    D12 = np.zeros((T.shape))
    D13 = np.zeros((T.shape))

    sint = np.sin(theta)

    for i in range(len(r)):
        for j in range(len(theta)):
            for k in range(len(phi)):
                D12[i, j, k] = (1/r[i])* DTt[i, j, k]
                D13[i, j, k] = (1/(r[i]*np.sin(theta[j])))* DTp[i, j, k]
    return  DTr, D12, D13, DTt, DTp

#################### calculations

def snapshot_nonline_computation(filenameu, filenameb, filenamet, lcut=None, mcut=None, path = None, save_path = None, time_stamp=None):

    filenameU = filenameu
    filenameB = filenameb
    filenameT = filenamet

    irs, ire, r, theta, phi, theta_tr, phi_tr, Ur, Ut, Up, Utrr, Utrt, Utrp = load_fieds(path + filenameU, lcut, mcut)
    _,_,_,_,_,_,_, Br, Bt, Bp, Btrr, Btrt, Btrp 	  = load_fieds(path + filenameB, lcut, mcut)

    Br, Bt, Bp = Br[irs:ire+1], Bt[irs:ire+1], Bp[irs:ire+1]
    Btrr, Btrt, Btrp = Btrr[irs:ire+1], Btrt[irs:ire+1], Btrp[irs:ire+1]

    lmax, mmax, mres, r, theta, phi, theta_tr, phi_tr, T, Ttr, time1 = load_fieds_scalar(path + filenameT,lcut, mcut)

    TDV1, TDV2, TDV3, TDV4, TDV5, TDV6, TDV7, TDV8, TDV9, DV1, DV2, DV3, DV4, DV5, DV6, DV7, DV8, DV9 = grad_vector(Utrr, Utrt, Utrp, r, theta_tr, phi_tr)
    TDB1, TDB2, TDB3, TDB4, TDB5, TDB6, TDB7, TDB8, TDB9, DB1, DB2, DB3, DB4, DB5, DB6, DB7, DB8, DB9 = grad_vector(Btrr, Btrt, Btrp, r, theta_tr, phi_tr)
    TDT1, TDT2, TDT3, DTt, DTp           = grad_scalar(Ttr, r, theta_tr, phi_tr)

    uur, uut, uup = non_lin_same_vector(Ur, Ut, Up, r, theta, phi)
    bbr, bbt, bbp = non_lin_same_vector(Br, Bt, Bp, r, theta, phi)

    ubr, ubt, ubp = non_lin_diff_vector(Ur, Ut, Up, Br, Bt, Bp, r, theta, phi, 'uGb')
    bur, but, bup = non_lin_diff_vector(Ur, Ut, Up, Br, Bt, Bp, r, theta, phi, 'bGu')

    UT   = non_lin_temp(Ur, Ut, Up, T, r, theta, phi)
    UTtr = non_lin_temp(Utrr, Utrt, Utrp, Ttr, r, theta_tr, phi_tr)

    uutrr, uutrt, uutrp = non_lin_same_vector(Utrr, Utrt, Utrp, r, theta_tr, phi_tr)
    bbtrr, bbtrt, bbtrp = non_lin_same_vector(Btrr, Btrt, Btrp, r, theta_tr, phi_tr)
    
    ubtrr, ubtrt, ubtrp = non_lin_diff_vector(Utrr, Utrt, Utrp, Btrr, Btrt, Btrp, r, theta_tr, phi_tr, 'uGb')
    butrr, butrt, butrp = non_lin_diff_vector(Utrr, Utrt, Utrp, Btrr, Btrt, Btrp, r, theta_tr, phi_tr, 'bGu')

    save_npy(r, theta, phi, Utrr, Utrt, Utrp, Btrr, Btrt, Btrp,'{0:}/UB_trunc_{1:}'.format(save_path, time_stamp))
    save_npy_scalar(TDT1, TDT2, TDT3, DTt, DTp, Ttr, UT, UTtr,'{0:}/TTr_{1:}'.format(save_path, time_stamp))

    save_npy(TDV1, TDV2, TDV3, TDV4, TDV5, TDV6, TDV7, TDV8, TDV9,'{0:}/TDVF_grad_{1:}'.format(save_path, time_stamp))
    save_npy(TDB1, TDB2, TDB3, TDB4, TDB5, TDB6, TDB7, TDB8, TDB9,'{0:}/TDBF_grad_{1:}'.format(save_path, time_stamp))
    save_npy(DV1, DV2, DV3, DV4, DV5, DV6, DV7, DV8, DV9,'{0:}/TDVF_deri_{1:}'.format(save_path, time_stamp))
    save_npy(DB1, DB2, DB3, DB4, DB5, DB6, DB7, DB8, DB9,'{0:}/TDBF_deri_{1:}'.format(save_path, time_stamp))

    nonlin_full  = [uur, uut, uup, bbr, bbt, bbp, ubr, ubt, ubp, bur, but, bup]
    nonlin_trunc  = [uutrr, uutrt, uutrp, bbtrr, bbtrt, bbtrp, ubtrr, ubtrt, ubtrp, butrr, butrt, butrp]
    dim = [r, theta, phi]
    para = [lmax, mmax, mres]

    tempra = [UT, UTtr]

    return nonlin_full, nonlin_trunc, dim, time1, para, tempra

