from pizza import PizzaFields, costf, equatContour
from scipy.integrate import simps
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as S

f = PizzaFields()

gold = f.radius[::-1]
nnew = 1024
gnew = np.linspace(gold[0], gold[-11], nnew)
usinterp = np.zeros((f.n_phi_max, len(gnew)), np.float64)
upinterp = np.zeros((f.n_phi_max, len(gnew)), np.float64)
print(len(gnew))
# Interpolate radial grid
for i in range(f.us.shape[0]):
    val = f.us[i, ::-1]
    tckp = S.splrep(gold, val)
    fnew = S.splev(gnew, tckp)
    usinterp[i, :] = fnew

    val = f.uphi[i, ::-1]
    tckp = S.splrep(gold, val)
    fnew = S.splev(gnew, tckp)
    upinterp[i, :] = fnew

ushat = np.fft.fft2(usinterp)/f.n_phi_max/len(gnew)
ushat = np.fft.fftshift(ushat, axes=(0,1))
uphat = np.fft.fft2(upinterp)/f.n_phi_max/len(gnew)
up0 = np.fft.fftshift(uphat[0, :])
uphat = np.fft.fftshift(uphat, axes=(0,1))
fr = np.pi * np.fft.fftshift(np.fft.fftfreq(len(gnew), d=gnew[1]-gnew[0]))
print(fr.max(), fr.min())
dphi = 2. * np.pi / (f.n_phi_max)
fm = 2.* np.pi * np.fft.fftshift(np.fft.fftfreq(f.n_phi_max, d=dphi))
print(fm.max(), fm.min())

fig0 = plt.figure()
ax0 = fig0.add_subplot(111)
ax0.loglog(fr, abs(up0)**2)

nrj = abs(ushat)**2 + abs(uphat)**2

fig = plt.figure()
ax = fig.add_subplot(111)
levs = np.linspace(13, 20, 17)
vmax = np.log10(nrj).max()-2.
vmin = vmax-4.
im = ax.pcolormesh(fm, fr, np.log10(nrj.T), vmax=vmax, vmin=vmin,
                   cmap=plt.get_cmap('magma'))

alpha = np.linspace(0., 2*np.pi, 128)

vpmean = abs(f.uphi.mean(axis=0))
beta = -f.radius/(f.radius[0]**2-f.radius**2) * 2. / f.ek
kbeta = vpmean[1:]/abs(beta[1:])
kbm = abs(simps(kbeta*f.radius[1:], f.radius[1:]) / simps(f.radius, f.radius))
kbeta = 1./np.sqrt(kbm) / 2. /np.pi

fac = kbeta # from average of np.sqrt{uphi/beta} ?
ax.plot(fac*np.cos(alpha)*np.sqrt(np.cos(alpha)),
        fac*np.sin(alpha)*np.sqrt(np.cos(alpha)), 'k-')
ax.plot(-fac*np.cos(alpha)*np.sqrt(np.cos(alpha)),
        -fac*np.sin(alpha)*np.sqrt(np.cos(alpha)), 'k-')
fig.colorbar(im)
fig.tight_layout()
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)

plt.show()

