import numpy as np
import shtns
from read_write import *
import numba

###################

def forward_transform(ur, us, ut, sh):
    Nlat, Nphi = sh.set_grid(nl_order=2, flags=shtns.sht_gauss)
    #print ("Nlat = \t", Nlat, "\t Nphi \t", Nphi)
    ur_lm, us_lm, ut_lm = sh.analys(ur, us, ut)
    return ur_lm, us_lm, ut_lm

def inverse_transform(ur_lm, us_lm, ut_lm, sh):
    ur, ut, up = sh.synth(ur_lm, us_lm, ut_lm)
    return ur, ut, up

def forward_transform_scalar(ur, sh):
    Nlat, Nphi = sh.set_grid(nl_order=2, flags=shtns.sht_gauss)
    ur_lm = sh.analys(ur)
    return ur_lm

def inverse_transform_scalar(ur_lm, sh):
    ur = sh.synth(ur_lm)
    return ur

##### truncate l and m

def truncate_l_m(Ar, At, Ap, lmax, mmax, lcut, mcut, mres, sh1, sh2):#, lmax, mmax):

    nlm = shtns.nlm_calc(lcut, mcut, mres)

#    print ("Shape trunc = \t", nlm)

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

def truncate_field_from_real(Ar,At, Ap, lmax, mmax, mres,  lcut, mcut, mode = None):

    if mode=='full': 
        sh1 = shtns.sht(lmax, mmax, mres)
        sh2 = shtns.sht(lcut, mcut, mres)
    else:
        sh1 = shtns.sht(lcut, mcut, mres)
        sh2 = sh1

    Nr = Ar.shape[0]

    Nlat, Nphi = sh2.set_grid(nl_order=2, flags=shtns.sht_gauss)

    Art = np.zeros((Nr, Nlat, Nphi))
    Att = np.zeros((Nr, Nlat, Nphi))
    Apt = np.zeros((Nr, Nlat, Nphi))

    for ir in range(Ar.shape[0]):

        ur = Ar[ir, :, : ]
        ut = At[ir, : , :]
        up = Ap[ir, : , :]

        if mode == 'full':
            #print ("Manohar")
            ur_lm, us_lm, ut_lm = forward_transform(ur, ut, up, sh1)
        else:
            ur_lm, us_lm, ut_lm = forward_transform(ur, ut, up, sh2)

        ur_r, ut_r, up_r = truncate_l_m(ur_lm, us_lm, ut_lm, lmax, mmax,  lcut, mcut, mres, sh1, sh2)

        Art[ir,:,:] = ur_r
        Att[ir,:,:] = ut_r
        Apt[ir,:,:] = up_r

    return Art, Att, Apt

def truncate_l_m_scalar(Ar, lmax, mmax, lcut, mcut, mres, sh1, sh2):#, lmax, mmax):
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

    Ur = sh2.synth(Atrr)

    return Ur

def truncate_scalar_field_from_real(Ar, lmax, mmax, mres,  lcut, mcut, mode = None):

    if mode=='full':
        sh1 = shtns.sht(lmax, mmax, mres)
        sh2 = shtns.sht(lcut, mcut, mres)
    else:
        sh1 = shtns.sht(lcut, mcut, mres)
        sh2 = sh1

    Nr = Ar.shape[0]

    Nlat, Nphi = sh2.set_grid(nl_order=2, flags=shtns.sht_gauss)

    Art = np.zeros((Nr, Nlat, Nphi))

    for ir in range(Ar.shape[0]):
        ur = Ar[ir, :, : ]
        if mode == 'full':
            ur_lm = forward_transform_scalar(ur, sh1)
        else:
            ur_lm = forward_transform_scalar(ur, sh2)

        ur_r = truncate_l_m_scalar(ur_lm, lmax, mmax,  lcut, mcut, mres, sh1, sh2)

        Art[ir,:,:] = ur_r

    return Art

def comput_truncated_nonlinear(A, B, C, D, para, lcut=None, mcut=None, data_path=None, save_path=None, time_stamp=None):

    lmax = para[0]
    mmax = para[1]
    mres = para[2]

    r = C[0]; theta = C[1]; phi = C[2]

    uur = A[0];  uut = A[1];  uup = A[2]
    bbr = A[3];  bbt = A[4];  bbp = A[5]
    ubr = A[6];  ubt = A[7];  ubp = A[8]
    bur = A[9];  but = A[10]; bup  = A[11]

    uutrr = B[0];  uutrt = B[1];  uutrp = B[2]
    bbtrr = B[3];  bbtrt = B[4];  bbtrp = B[5]
    ubtrr = B[6];  ubtrt = B[7];  ubtrp = B[8]
    butrr = B[9];  butrt = B[10]; butrp = B[11]

    UT = D[0]
    UTtr = D[1]

 #   print ("shape main loop= \t", uur.shape)


    Ftuur, Ftuut, Ftuup = truncate_field_from_real(uur, uut, uup, lmax, mmax, mres, lcut, mcut, mode = 'full')
    Ftbbr, Ftbbt, Ftbbp = truncate_field_from_real(bbr, bbt, bbp, lmax, mmax, mres, lcut, mcut, mode = 'full')
    Ftubr, Ftubt, Ftubp = truncate_field_from_real(ubr, ubt, ubp, lmax, mmax, mres, lcut, mcut, mode = 'full')
    Ftbur, Ftbut, Ftbup = truncate_field_from_real(bur, but, bup, lmax, mmax, mres, lcut, mcut, mode = 'full')

    FtUT = truncate_scalar_field_from_real(UT, lmax, mmax, mres, lcut, mcut, mode = 'full')

    Fuur = Ftuur - uutrr; Fuut = Ftuut - uutrt; Fuup = Ftuup - uutrp
    Fbbr = Ftbbr - bbtrr; Fbbt = Ftbbt - bbtrt; Fbbp = Ftbbp - bbtrp
    Fubr = Ftubr - ubtrr; Fubt = Ftubt - ubtrt; Fubp = Ftubp - ubtrp
    Fbur = Ftbur - butrr; Fbut = Ftbut - butrt; Fbup = Ftbup - butrp

    FUT = FtUT - UTtr

    TFuur, TFuut, TFuup = truncate_field_from_real(Fuur, Fuut, Fuup, lmax, mmax, mres, lcut, mcut)
    TFbbr, TFbbt, TFbbp = truncate_field_from_real(Fbbr, Fbbt, Fbbp, lmax, mmax, mres, lcut, mcut)
    TFubr, TFubt, TFubp = truncate_field_from_real(Fubr, Fubt, Fubp, lmax, mmax, mres, lcut, mcut)
    TFbur, TFbut, TFbup = truncate_field_from_real(Fbur, Fbut, Fbup, lmax, mmax, mres, lcut, mcut)

    TFUT = truncate_scalar_field_from_real(FUT, lmax, mmax, lcut, mcut)

    save_npy(r, theta, phi, TFuur, TFuut, TFuup, TFbbr, TFbbt, TFbbp, '{0:}/overline_tilde_full_AA_{1:}'.format(save_path, time_stamp))
    save_npy(r, theta, phi, TFubr, TFubt, TFubp, TFbur, TFbut, TFbup, '{0:}/overline_tilde_full_AB_{1:}'.format(save_path, time_stamp))

    save_npy_scalar_two(FUT, TFUT,'{0:}/overline_tilde_UT{1:}'.format{save_path, time_stamp})

    save_npy(r, theta, phi, uutrr, uutrt, uutrp, bbtrr, bbtrt, bbtrp, '{0:}/trunc_field_AA_{1:}'.format(save_path, time_stamp))
    save_npy(r, theta, phi, ubtrr, ubtrt, ubtrp, butrr, butrt, butrp, '{0:}/trunc_field_AB_{1:}'.format(save_path, time_stamp))

    return 0
	
