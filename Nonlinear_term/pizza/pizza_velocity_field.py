import sys
import matplotlib.pylab as plt

sys.path.append('/nfs_scratch/sharmam/Nonlinear_term/pizza/pizza-master/python/')

from pizza import *

f = PizzaFields(tag='E1e8Ra5e12I', ivar=1, datadir = '/nfs_scratch/schaeffn/data_pizza/E1e8Ra5e12T2D/')


r = f.radius

# ~ f1 = PizzaRadial()

print ("velocity r field = \t", f.us.shape)
print ("velocity phi field = \t", f.uphi.shape)
print ("tem field = \t", f.temp.shape)

print ("radius of =\t",f.radius.shape)
print ("nr  =\t",f.n_r_max)
print ("n max =\t",f.n_m_max)
print ("n phi =\t",f.n_phi_max)
print ("time =\t",f.time)
print ("radius =\t",np.min(r))
print ("radius =\t",np.max(r))
# ~ print ("radius diff =\t",r[1:] - r[:-1])

x = np.arange(len(r))

plt.plot(x, r)
plt.show()
