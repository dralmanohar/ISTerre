import numpy as np



def destandrization(path, name):
	a = np.load(path + '/' + 'r_comp/notebooks/model_unet/' + name)
	b = np.load(path + '/' + 'r_comp/notebooks/model_unet/' + 'mean_output.npy')
	c = np.load(path + '/' + 'r_comp/notebooks/model_unet/' + 'std_output.npy')
	
	d = a*c + b
	
	return d[0,0]

ut_00_32_t = destandrization('0_32', 'label_train.npy')
ut_32_64_t = destandrization('32_64', 'label_train.npy')
ut_64_96_t = destandrization('64_96', 'label_train.npy')
ut_96_128_t = destandrization('96_128', 'label_train.npy')
ut_128_152_t = destandrization('128_152', 'label_train.npy')

ut_00_32_p = destandrization('0_32',  'preds_train.npy')
ut_32_64_p = destandrization('32_64', 'preds_train.npy')
ut_64_96_p = destandrization('64_96', 'preds_train.npy')
ut_96_128_p = destandrization('96_128', 'preds_train.npy')
ut_128_152_p = destandrization('128_152', 'preds_train.npy')



print ("ut_0_32 = \t",ut_32_64_t.shape)


ut_true = np.zeros((32, 152, 256))
ut_preds = np.zeros((32, 152, 256))


for i in range(32):
	for j in range(152):
		
		if 0<=j<32:
			ut_true[i, j]  = ut_00_32_t[i,j]
			ut_preds[i, j] = ut_00_32_p[i,j]
		elif 32<=j<64:
			j1 = j - 32
			ut_true[i, j]  = ut_32_64_t[i,j1]
			ut_preds[i, j] = ut_32_64_p[i,j1]
		elif 64<=j<96:
			j1 = j - 64
			ut_true[i, j]  = ut_64_96_t[i,j1]
			ut_preds[i, j] = ut_64_96_p[i,j1]
		elif 96<=j<128:
			j1 = j - 96
			ut_true[i, j]  = ut_96_128_t[i,j1]
			ut_preds[i, j] = ut_96_128_p[i,j1]
		elif 128<=j<152:
			j1 = j - 128
			ut_true[i, j]   = ut_96_128_t[i,j1]
			ut_preds[i, j]  = ut_96_128_p[i,j1]

np.save("ut_true.npy",ut_true)
np.save("ut_preds.npy",ut_preds)
			
# ~ print ("shape dns = \t", ut_co)

print ("shape ut \t",ut_true.shape)


