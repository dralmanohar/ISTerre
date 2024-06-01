import numpy as np
import os



def destandrization(path, name):
	a = np.load(path +  name)
	b = np.load(path + 'mean_output.npy')
	c = np.load(path + 'std_output.npy')
	
	d = a*c + b
	
	return d[0,0]

path = './'
number_folder = os.listdir(path)

print (number_folder)


number_folder.remove('To_get_full_field_stack.py')
number_folder.remove('outlog')
number_folder.remove('f_uu')
number_folder.remove('f_ub')
number_folder.remove('f_bu')
number_folder.remove('f_bb')

print (number_folder)

path_all_fields = []

for i in range(len(number_folder)):
    path_sub = os.listdir(number_folder[i])

    path_sub.remove('evaluate_ML.py')
    path_sub.remove('path.txt')
    path_sub.remove('plot_in_sphere.py')
    path_sub.remove('plot_nonliner_real_space.py')
    path_sub.remove('read_write.py')
    path_sub.remove('UB_trunc_0299.npy')
    path_sub.remove('__pycache__')
    path_sub.remove('To_get_full_field_stack.py')
    #path_sub.remove('outlog')
    
    print ("path_sub = \t", path_sub)

    for sub_dir in path_sub:

        sub_dir_path = path + '/' + number_folder[i] + '/' + sub_dir

        path_field = os.listdir(sub_dir_path)

        for field_dir  in path_field:

            path_model = sub_dir_path + '/' + field_dir + '/' + 'notebooks' + '/' + 'model_unet'

            path_all_fields.append(path_model)
            #print ("path field = \t", path_model )


data_path = []
field_comp = []
data_field = []

for file in path_all_fields:

    file_name = file.split("/")

    data_path.append(file_name[2])
    data_field.append(file_name[3])
    field_comp.append(file_name[4])

data_path = np.unique(data_path)
data_field = np.unique(data_field)
field_comp = np.unique(field_comp)

#ut_true = np.zeros((32, 152, 256))
#ut_preds = np.zeros((32, 152, 256))

for field in data_field:
	
    for comp in field_comp:

        ut_true = np.zeros((32, 152, 256))
        ut_preds = np.zeros((32, 152, 256))

        for data in data_path:

            for path in path_all_fields:

                data_path_load = path.split("/")

                #print (data_path_load)

                folder = data_path_load[2]
                non_field = data_path_load[3]
                non_comp = data_path_load[4]

                if data != folder:
                    if field==non_field:
                        if comp==non_comp:

                            path_data_load_field = path + '/'# + folder #+ non_field + '/' + comp

                            region = folder.split("_")
                            #print ("region = \t", region)
                            region1 = int(region[0].strip())
                            region2 = int(region[1].strip())

                            ut_model = destandrization(path_data_load_field, 'label_train.npy')
                            up_model = destandrization(path_data_load_field, 'preds_train.npy')



                            for i in range(32):
                                  for j in range(152):
  #                                  print ("shape = \t", ut_model.shape, "\t ut_true = \t", ut_true.shape, "\t ut preds = \t", ut_preds.shape, "\t up_model = \t", up_model.shape)
                                    #print ("i = \t", i, "\t region = \t", region1, "\t region 2 = \t", region2)
                                    
                                    if region1<=0 and region2<=32:
                                        if region1<=j<region2:

                                            ut_true[i, j, :]  = ut_model[i,j, :]
                                            ut_preds[i, j, :] = up_model[i,j, :]

                                    elif region1<=32 and region2<=64:
                                        if region1<=j<region2:

                                 #           print ("i = \t", i,"\t j = \t",j, "\t region = \t", region1, "\t region 2 = \t", region2)

                                            j1 = j - 32
                                            ut_true[i, j]  = ut_model[i,j1]
                                            ut_preds[i, j] = up_model[i,j1]
                                            
                                    elif region1<=64 and region2<=96:
                                        if region1<=j<region2:
                                            j1 = j - 64
                                            ut_true[i, j]  = ut_model[i,j1]
                                            ut_preds[i, j] = up_model[i,j1]

                                    elif region1<=96 and region2<=128:
                                        if region1<=j<region2:
                                            j1 = j - 96
                                            ut_true[i, j]  = ut_model[i,j1]
                                            ut_preds[i, j] = up_model[i,j1]
                                    
                                    elif region1<=128 and region2<=152:
                                        if region1<=j<region2:
                                            j1 = j - 128
                                            ut_true[i, j]   = ut_model[i,j1]
                                            ut_preds[i, j]  = up_model[i,j1]
        
        path_dir = './'
        MYDIR = path_dir + '/' + '{0:}/{1:}'.format(field, comp)
        CHECK_FOLDER = os.path.isdir(MYDIR)

        # If folder doesn't exist, then create it.

        if not CHECK_FOLDER:
            os.makedirs(MYDIR)
            print("created folder : ", MYDIR)

        else:
            print(MYDIR, "folder already exists.")
        
        print ("MYDIR = \t", MYDIR, "\t path = \t", path_dir)#, "\t field = \t ", field, "\t comp = \t", comp)
        np.save("%s/ut_true.npy"%(MYDIR),ut_true)
        np.save("%s/ut_preds.npy"%(MYDIR),ut_preds)
    

    #print (field)

#for path in path_model:

 #   data_path = path.split("/")

  #  if 


print ("data path = \t", data_path)
print ("data field = \t", data_field)
print ("field comp = \t", field_comp)


#print ("data_path = \t",  np.unique(data_path))
#print ("data_field = \t", data_field)
#print ("field_comp = \t", field_comp)

    #print ("path model = \t", file_name[2])

'''
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
'''

