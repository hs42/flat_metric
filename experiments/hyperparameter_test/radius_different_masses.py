import json
import numpy as np
import subprocess
import os


'''
quick / most important parameters
'''
dim = 4
few_samples = True


if few_samples:
    sample_size_factor =  2**dim 
else:
    sample_size_factor = 6 * 2**dim #account for need for more training data in higher dimensions



#radii_to_test = list(np.arange(0.01, 3, 0.3)) + list(range(3,18,5)) #list concatenation
radii_to_test = list(np.arange(0.01, 3, 0.3)) + list(range(3,13,1)) #list concatenation


linear_layer_type = 'spectral_normal'

path_to_default =      'lnets{s}tasks{s}dualnets{s}configs{s}default_2_diracs.json'.format(s=os.sep)
config_To_be_written = 'lnets{s}tasks{s}dualnets{s}configs{s}2_diracs.json'.format(s=os.sep)
out_path = 'out{s}Test_hyperparameters_many_radii{s}'.format(d=dim, s=os.sep, n=['many', 'few'][few_samples])

#read data
with open(path_to_default) as f:
   data = json.load(f)

#correct center_x if dimensionality does not match
if len(data['distrib1']['center_x']) == 1 and dim > 1:
    data['distrib1']['center_x'] = dim*data['distrib1']['center_x']#repeat value in list
elif len(data['distrib1']['center_x']) != dim:
    print('Error: give enough coordinates for center_x in distrib1')
if len(data['distrib2']['center_x']) == 1 and dim > 1:
    data['distrib2']['center_x'] = dim*data['distrib2']['center_x'] #repeat value in list
elif len(data['distrib2']['center_x']) != dim:
    print('Error: give enough coordinates for center_x in distrib2')

#set certain parameters



m = int(sample_size_factor*10)
n = int(sample_size_factor*50)

data['distrib1']['dim'] = dim 
data['distrib2']['dim'] = dim

data['distrib1']['sample_size'] = int(m)
data['distrib2']['sample_size'] = int(n)
data['distrib1']['test_sample_size'] = max(1,int(0.1*m))
data['distrib2']['test_sample_size'] = max(1,int(0.1*n))


epochs = 2000
layers = []
groupings = []
activation = 'group_sort'
#enter loop and sweep through different values for the radius
N_test = 6  
for i in range(N_test):

    if i == 0:
        #this is the default (=control) case
        linear_layer_type = 'spectral_normal'
        layers = [128, 128, 1]
        groupings = [2, 2, 1]
    elif i == 1:
        #compare to Björck orthonormalization
        linear_layer_type = 'bjorck'
        layers = [128, 128, 1]
        groupings = [2, 2, 1]
    elif i == 2:
        #make net deeper, add more training time
        linear_layer_type = 'spectral_normal'
        layers = [128, 128, 128, 128, 128, 1]
        groupings = [2, 2, 2, 2, 2, 1]
        epochs = 10000
    elif i == 3:
        #make layers bigger
        linear_layer_type = 'spectral_normal'
        layers = [512, 512, 1]
        groupings = [8, 8, 1]
        epochs = 10000
    elif i == 4:
        #make net deeper, add more training time, switch to Bjorck
        linear_layer_type = 'bjorck'
        layers = [128, 128, 128, 128, 128, 1]
        groupings = [2, 2, 2, 2, 2, 1]
        epochs = 10000
    elif i == 5:
        #make layers bigger, switch to Bjorck
        linear_layer_type = 'bjorck'
        layers = [512, 512, 1]
        groupings = [8, 8, 1]
        epochs = 10000


    data['model']['linear']['type'] = linear_layer_type
    data['model']['layers'] = layers
    data['model']['groupings'] = groupings
    data['model']['activation'] = activation
    data['optim']['epochs'] = epochs


    out_path2 = os.path.join(out_path, str(i))
    data['output_root'] = out_path2

    for r in radii_to_test:
        data['distrib2']['radius'] = r

        #write new config file
        with open(config_To_be_written, "w") as write_file:
            json.dump(data, write_file)

        #compute flat metric
        subprocess.call("python lnets{s}tasks{s}dualnets{s}mains{s}train_dual.py ".format(s=os.sep) + config_To_be_written, shell=True)


