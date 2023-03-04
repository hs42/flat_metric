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



#radii_to_test = np.arange(radius_start, radius_stop, radius_step)
radii_to_test = list(np.arange(0.01, 3, 0.3)) + list(range(3,18,5)) #list concatenation
m_to_test = sample_size_factor*np.array([5, 10, 20], dtype=int)
n_to_test = sample_size_factor*np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=int)

linear_layer_type = 'spectral_normal'

path_to_default =      'lnets{s}tasks{s}dualnets{s}configs{s}default_2_diracs.json'.format(s=os.sep)
config_To_be_written = 'lnets{s}tasks{s}dualnets{s}configs{s}2_diracs.json'.format(s=os.sep)
out_path = 'out{s}Probe_radius_different_masses{s}{n}_samples{s}dimension_{d}'.format(d=dim, s=os.sep, n=['many', 'few'][few_samples])

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



data['output_root'] = out_path

#enter loop and sweep through different values for the radius
for m in m_to_test:
    for n in n_to_test:
        for r in radii_to_test:
            data['distrib2']['radius'] = r

            data['distrib1']['sample_size'] = int(m)
            data['distrib2']['sample_size'] = int(n)
            data['distrib1']['test_sample_size'] = max(1,int(0.1*m))
            data['distrib2']['test_sample_size'] = max(1,int(0.1*n))

            #write new config file
            with open(config_To_be_written, "w") as write_file:
                json.dump(data, write_file, indent=4)

            #compute flat metric
            subprocess.call("python lnets{s}tasks{s}dualnets{s}mains{s}train_dual.py ".format(s=os.sep) + config_To_be_written, shell=True)


