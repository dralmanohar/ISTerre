import numpy as np
import os
import h5py

from read_write import*
import glob
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

def load_numpy(filename):
	
	filename = filename
	
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


def load_numpy_scalar(filename):

    filename = filename
    data = np.load(filename, allow_pickle=True)

    r         = data.item().get('r')
    theta     = data.item().get('theta')
    phi       = data.item().get('phi')
    Tr  	  = data.item().get('Tr')
    Ttrr 	  = data.item().get('Trr')
    deri   	  = data.item().get('deri')
    nlin      = data.item().get('nlin')
    nlintr    = data.item().get('nlintr')

    return r, theta, phi, Tr, Ttrr, deri, nlin, nlintr

def data_generator_frame(path):
    my_list = sorted(os.listdir(data_path))
    sample = 0

    print ("my;ist = \t",len(my_list))

    number_field = len(my_list)

    train = int(number_field*0.886)
    test  = int(number_field*0.103)
    val   = int(number_field*0.011)

    print ("train = \t",train)

    for folder in (my_list):
        path = data_path + folder

        file_name1 = path + '/' + 'UB_trunc_%s.npy'%(folder)
        file_name2 = path + '/' + 'TTr_%s.npy'%(folder)

        file_name3 = path + '/' + 'TDVF_grad_%s.npy'%(folder)
        file_name4 = path + '/' + 'TDBF_grad_%s.npy'%(folder)

        file_name5 = path + '/' + 'overline_tilde_full_AA_%s.npy'%(folder)
        file_name6 = path + '/' + 'overline_tilde_full_AB_%s.npy'%(folder)

        file_name7 = path + '/' + 'trunc_field_AA_%s.npy'%(folder)
        file_name8 = path + '/' + 'trunc_field_AB_%s.npy'%(folder)

        r, theta, phi, Ur, Ut, Up, Br, Bt, Bp = load_numpy(file_name1)
        DT1,_,_,DT2, DT3,    Tr,_,UTtr		  = load_numpy_scalar(file_name2)

        DV11, DV12, DV13, DV21, DV22, DV23, DV31, DV32, DV33 = load_numpy(file_name3)
        DB11, DB12, DB13, DB21, DB22, DB23, DB31, DB32, DB33 = load_numpy(file_name4)

        _, _, _, UUr, UUt, UUp, BBr, BBt, BBp  = load_numpy(file_name5)
        _, _, _, UBr, UBt, UBp, BUr, BUt, BUp  = load_numpy(file_name6)


        _,_,_,   UUtrr, UUtrt, UUtrp, BBtrr, BBtrt, BBtrp = load_numpy(file_name7)
        _,_,_,  UBtrr, UBtrt, UBtrp, BUtrr, BUtrt, BUtrp  = load_numpy(file_name8)

        U = {'Ur':Ur, 'Ut': Ut, 'Up':Up}
        B = {'Br':Br, 'Bt': Bt, 'Bp':Bp}
        T = {'Tr':Tr}

        DV = {'DV11':DV11, 'DV12':DV12, 'DV13':DV13, 'DV21':DV21, 'DV22':DV22, 'DV23':DV23, 'DV31':DV31, 'DV32':DV32, 'DV33':DV33}
        DB = {'DB11':DB11, 'DB12':DB12, 'DB13':DB13, 'DB21':DB21, 'DB22':DB22, 'DB23':DB23, 'DB31':DB31, 'DB32':DB32, 'DB33':DB33}
        DT = {'DT1':DT1, 'DT2':DT2, 'DT3':DT3, 'UTtr':UTtr}

        UU = {'UUr':UUr, 'UUt':UUt, 'UUp':UUp}
        BB = {'BBr':BBr, 'BBt':BBt, 'BBp':BBp}
        UB = {'UBr':UBr, 'UBt':UBt, 'UBp':UBp}
        BU = {'BUr':BUr, 'BUt':BUt, 'BUp':BUp}

        UUtr = {'UUtrr':UUtrr, 'UUtrt':UUtrt, 'UUtrp':UUtrp}
        BBtr = {'BBtrr':BBtrr, 'BBtrt':BBtrt, 'BBtrp':BBtrp}
        UBtr = {'UBtrr':UBtrr, 'UBtrt':UBtrt, 'UBtrp':UBtrp}
        BUtr = {'BUtrr':BUtrr, 'BUtrt':BUtrt, 'BUtrp':BUtrp}

        print ("max UUR = \t",np.amax(UUtrr))
        print ("max BBtr = \t", np.amax(BBtrr))
        print ("max UBtr = \t", np.amax(UBtrr))
        print ("max BUtr = \t",np.amax(BUtrr))


        fields = [U, B, T, DV, DB, DT, UU, BB, UB, BU, UUtr, BBtr, UBtr, BUtr]

        if sample<train:
            path = data_path
            dire = make_direcotry(path, 'train')
            sample +=1
        
        elif train<=sample<test + train:
            path = data_path
            dire = make_direcotry(path, 'test')
            sample +=1
        else:
            path = data_path
            dire = make_direcotry(path, 'val')
            sample +=1

        number_str = str(sample)
        zero_filled_number = number_str.zfill(4)

        for name in fields:
            for keys, values in name.items():
                comp = keys[-1]
                variable = keys[:-1]
                filename = dire + '/' + variable + '_{0:}_{1:}'.format(comp, zero_filled_number) + '.h5'
                field = values[50-16:50+16, :, :]
                write_3D(filename, field)
    return 0


##############

if __name__ == '__main__':
	data_path = '/nfs_scratch/sharmam/Nonlinear_term/results/3D_nathanael/test/'
	# ~ data_path = '../../result/test/'
	data_generator_frame(data_path)
