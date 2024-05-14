import numpy as np
import os
import h5py
import glob
from read_write import*
######### filenames

def make_direcotry(path, com):
	MYDIR = ("{0:}/{1:}".format(path, com))
	CHECK_FOLDER = os.path.isdir(MYDIR)
	
	if not CHECK_FOLDER:
		os.makedirs(MYDIR)
		print("created folder : ", MYDIR)
	else:
		print(MYDIR, "folder already exists.")
	
	return MYDIR
	

	var_dict = {'r':r,'theta': theta, 'ur': ur, 'ut': ut, 'temp':temp, 'durr':durr, 'durt': durt, 'dutr': dutr, 'dutt': dutt, 'dTr': dTr, 'dTt': dTt, 'nlinr': nlinr, 'nlint': nlint}
	
def load_numpy(filename):

    filename = filename
    data = np.load(filename, allow_pickle=True)
    r = data.item().get('r')
    theta = data.item().get('theta')
    ur = data.item().get('ur')
    ut = data.item().get('ut')
    temp = data.item().get('temp')
    durr = data.item().get('durr')
    durt = data.item().get('durt')
    dutr = data.item().get('dutr')
    dutt = data.item().get('dutt')
    dTr = data.item().get('dTr')
    dTt = data.item().get('dTt')
    nlinr = data.item().get('nlinr')
    nlint = data.item().get('nlint')
    nlintr_r = data.item().get('nlintr_r')
    nlintr_t = data.item().get('nlintr_t')
    fububr   = data.item().get('fububr')
    fububt   = data.item().get('fububt')
    tfur     = data.item().get('tfur')
    tfut     = data.item().get('tfut')
    temp_d   = data.item().get('temp_d')
    ttemp_d  = data.item().get('ttemp_d')
    nlinT    = data.item().get('ont')

    return r, theta, ur, ut, temp, durr, durt, dutr, dutt, dTr, dTt, nlinr, nlint, nlintr_r, nlintr_t, fububr, fububr, ttemp_d, nlinT


def data_generator_frame(path, save_path = None):

    files = folders = 0

    for _, dirnames, filenames in os.walk(path):
        files += len(filenames)
        folders += len(dirnames)

    my_list = sorted(os.listdir(data_path))

    my_list_u = []
    my_list_l = []

    for letter in my_list:
        if letter[-1].isupper():
            my_list_u.append(letter)
        else:
            my_list_l.append(letter)


    my_list_final = sorted(my_list_l) + sorted(my_list_u)

    train = int(files*0.82)
    test  = int(files*0.179)
    val   = int(files*0.001)

    sample = 0
    
    for folder in (my_list_final):
        path = data_path + '/' + folder
        my_list_1 = sorted(os.listdir(path))


        for name in my_list_1:
            path1 = path + '/' + name

            files = os.listdir(path1)

            files = sorted(files)

            for file_name in files:

                file_name1 = path1 + '/' + file_name

                r, theta, ur, ut, temp, durr, durt, dutr, dutt, dTr, dTt, nlinr, nlint, nlintr_r, nlintr_t, fububr, fububt, nlinT_tr, Fnlintr = load_numpy(file_name1)

                U = {'ur':ur, 'ut': ut}
                T = {'T':temp}
                DV = {'durr':durr, 'durt':durt, 'dutr':dutr, 'dutt':dutt}
                DT = {'dTr':dTr, 'dTt':dTt}
                UU = {'nlinr':nlinr, 'nlint':nlint,'nlintr_r':nlintr_r, 'nlintr_t':nlintr_t, 'fububr':fububr, 'fububt':fububt, 'nlinT_tr':nlinT_tr, 'Fnlintr':Fnlintr}
                fields = [U, T, DV, DT, UU]#, T]

                print ("max fububr = \t", np.amax(fububr))
                print ("min fububr = \t", np.amin(fububr))
                print ("max fububt = \t", np.amax(fububr))
                print ("min fububt = \t", np.amin(fububt))

                if sample<train:
                    dire = make_direcotry(save_path, 'train')
                    sample +=1
                elif train<=sample<test + train:
                    dire = make_direcotry(save_path, 'test')
                    sample +=1
                else:
                    dire = make_direcotry(save_path, 'val')
                    sample +=1

                print ("sample = \t",sample)

                number_str = str(sample)
                zero_filled_number = number_str.zfill(6)

                count = 0
                for name in fields:
                    for keys, values in name.items():
                        comp = keys[-1]
                        variable = keys[:-1]
                        filename = dire + '/' + variable + '_{0:}_{1:}'.format(comp, zero_filled_number) + '.h5'
                        field = values[:, 0:64]
                        write_2D(filename, field)
    print ("sample = \t",sample)

    return 0

##############

if __name__ == '__main__':
	data_path = '/nfs_scratch/sharmam/Nonlinear_term/results/pizza/full_data_size/128/'
	save_path = '/nfs_scratch/sharmam/Nonlinear_term/results/pizza/full_data_size/128_ML_data/'
	data_generator_frame(data_path, save_path)
