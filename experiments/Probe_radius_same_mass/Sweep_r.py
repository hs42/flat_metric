import json
import numpy as np
import subprocess
import os
import warnings

'''
quick / most important parameters
'''
dim = 5
few_samples = True



if few_samples:
    sample_size = 5 * 2**dim 
else:
    sample_size = 30 * 2**dim #account for need for more training data in higher dimensions

if sample_size > 40000: #some threshold
    warnings.warn('Too many samples. Numpy might not be able to handle that. Will reduce samples to 40000') 
    sample_size = 40000

#radii_to_test = np.arange(radius_start, radius_stop, radius_step)
radii_to_test = list(np.arange(0.01, 3, 0.1)) + list(range(3,33,5)) #list concatenation

linear_layer_type = 'spectral_normal'

path_to_default =      'lnets{s}tasks{s}dualnets{s}configs{s}default_2_diracs.json'.format(s=os.sep)
config_To_be_written = 'lnets{s}tasks{s}dualnets{s}configs{s}2_diracs.json'.format(s=os.sep)
out_path = 'out{s}Probe_radius_same_mass{s}{n}_samples{s}dimension_{d}'.format(d=dim, s=os.sep, n=['many', 'few'][few_samples])

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
data['model']['linear']['type'] = linear_layer_type

data['distrib1']['dim'] = dim 
data['distrib2']['dim'] = dim

data['distrib1']['sample_size'] = sample_size
data['distrib2']['sample_size'] = sample_size
data['distrib1']['test_sample_size'] = max(1,int(0.1*sample_size))
data['distrib2']['test_sample_size'] = max(1,int(0.1*sample_size))

data['output_root'] = out_path

#enter loop and sweep through different values for the radius

for r in radii_to_test:
    data['distrib2']['radius'] = r

    #write new config file
    with open(config_To_be_written, "w") as write_file:
        json.dump(data, write_file)

    #compute flat metric
    subprocess.call("python lnets{s}tasks{s}dualnets{s}mains{s}train_dual.py ".format(s=os.sep) + config_To_be_written, shell=True)


