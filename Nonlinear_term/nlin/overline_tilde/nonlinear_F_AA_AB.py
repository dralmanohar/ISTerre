import numpy as np
import shtns
from read_write import *
import numba

lmax, mmax = 100, 85
mres = 1

sh = shtns.sht(lmax, mmax)
Nlat, Nphi = sh.set_grid(nl_order=2, flags=shtns.sht_gauss)

###################

def forward_transform(ur, us, ut):
    ur_lm, us_lm, ut_lm = sh.analys(ur, us, ut)
    return ur_lm, us_lm, ut_lm

def inverse_transform(ur_lm, us_lm, ut_lm):
    ur, ut, up = sh.synth(ur_lm, us_lm, ut_lm)
    return ur, ut, up

def forward_transform_scalar(ur):
    ur_lm = sh.analys(ur)
    return ur_lm

def inverse_transform_scalar(ur_lm):
    ur = sh.synth(ur_lm)
    return ur


##### truncate l and m

def truncate_l_m(A, lmax, mmax, lcut, mcut):
    nlm = shtns.nlm_calc(lmax, mmax, mres)
    B = np.zeros(int(nlm), np.dtype('complex128'))
    l=0
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
    return B

def truncate_field_from_real(Ar,At, Ap, lcut, mcut):

    Nr = Ar.shape[0]

    Art = np.zeros((Ar.shape))
    Att = np.zeros((Ar.shape))
    Apt = np.zeros((Ar.shape))

    for ir in range(Ar.shape[0]):
        ur = Ar[ir, :, : ]
        ut = At[ir, : , :]
        up = Ap[ir, : , :]

        ur_lm, us_lm, ut_lm = forward_transform(ur, ut, up )

        ur_lmt = truncate_l_m(ur_lm, lmax, mmax, lcut, mcut)
        us_lmt = truncate_l_m(us_lm, lmax, mmax, lcut, mcut)
        ut_lmt = truncate_l_m(ut_lm, lmax, mmax, lcut, mcut)

        ur_r, ut_r, up_r = inverse_transform(ur_lmt, us_lmt, ut_lmt)

        Art[ir,:,:] = ur_r
        Att[ir,:,:] = ut_r
        Apt[ir,:,:] = up_r
    return Art, Att, Apt


def truncate_scalar_field_from_real(Ar, lcut, mcut):
    Nr = Ar.shape[0]

    Art = np.zeros((Ar.shape))

    for ir in range(Ar.shape[0]):
        ur = Ar[ir, :, : ]

        ur_lm = forward_transform_scalar(ur)

        ur_lmt = truncate_l_m(ur_lm, lmax, mmax, lcut, mcut)

        ur_r   = inverse_transform_scalar(ur_lmt)

        Art[ir,:,:] = ur_r

    return Art

def comput_truncated_nonlinear(A, B, C, D, lcut=None, mcut=None, data_path=None, save_path=None, time_stamp=None):

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

    Ftuur, Ftuut, Ftuup = truncate_field_from_real(uur, uut, uup, lcut, mcut)
    Ftbbr, Ftbbt, Ftbbp = truncate_field_from_real(bbr, bbt, bbp, lcut, mcut)
    Ftubr, Ftubt, Ftubp = truncate_field_from_real(ubr, ubt, ubp, lcut, mcut)
    Ftbur, Ftbut, Ftbup = truncate_field_from_real(bur, but, bup, lcut, mcut)

    FtUT = truncate_scalar_field_from_real(UT, lcut, mcut)

    Fuur = Ftuur - uutrr; Fuut = Ftuut - uutrt; Fuup = Ftuup - uutrp
    Fbbr = Ftbbr - bbtrr; Fbbt = Ftbbt - bbtrt; Fbbp = Ftbbp - bbtrp
    Fubr = Ftubr - ubtrr; Fubt = Ftubt - ubtrt; Fubp = Ftubp - ubtrp
    Fbur = Ftbur - butrr; Fbut = Ftbut - butrt; Fbup = Ftbup - butrp

    FUT  =  FtUT - UTtr

    TFuur, TFuut, TFuup = truncate_field_from_real(Fuur, Fuut, Fuup, lcut, mcut)
    TFbbr, TFbbt, TFbbp = truncate_field_from_real(Fbbr, Fbbt, Fbbp, lcut, mcut)
    TFubr, TFubt, TFubp = truncate_field_from_real(Fubr, Fubt, Fubp, lcut, mcut)
    TFbur, TFbut, TFbup = truncate_field_from_real(Fbur, Fbut, Fbup, lcut, mcut)

    TFUT  = truncate_scalar_field_from_real(FUT, lcut, mcut)

    save_npy(r, theta, phi, TFuur, TFuut, TFuup, TFbbr, TFbbt, TFbbp, '{0:}/overline_tilde_full_AA_{1:}'.format(save_path, time_stamp))
    save_npy(r, theta, phi, TFubr, TFubt, TFubp, TFbur, TFbut, TFbup, '{0:}/overline_tilde_full_AB_{1:}'.format(save_path, time_stamp))

    save_npy(r, theta, phi, Ftuur, Ftuut, Ftuup, Ftbbr, Ftbbt, Ftbbp,'{0:}/overline_AA_{1:}'.format(save_path, time_stamp))
    save_npy(r, theta, phi, Fuur, Fuut, Fuup, Fbbr, Fbbt, Fbbp,'{0:}/diff_AA_{1:}'.format(save_path, time_stamp))

    save_npy_scalar_two(FUT, TFUT,'{0:}/overline_UT{1:}'.format(save_path, time_stamp))

    save_npy(r, theta, phi, uutrr, uutrt, uutrp, bbtrr, bbtrt, bbtrp, '{0:}/trunc_field_AA_{1:}'.format(save_path, time_stamp))
    save_npy(r, theta, phi, ubtrr, ubtrt, ubtrp, butrr, butrt, butrp, '{0:}/trunc_field_AB_{1:}'.format(save_path, time_stamp))

    return 0
	
