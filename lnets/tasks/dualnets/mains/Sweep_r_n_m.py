import json
import numpy as np
import subprocess

'''
quick / most important parameters
'''
dim = 2

'''
radius_start = 0.1
radius_stop = 3.2
radius_step = 0.5

radii_to_test = np.arange(radius_start, radius_stop, radius_step)
'''


radii_to_test = [0.1, 0.5, 1.0, 1.5] + list(range(2,22,5)) #list concatenation

linear_layer_type = 'spectral_normal'

path_to_default =      './lnets/tasks/dualnets/configs/default_2_diracs.json'
config_To_be_written = './lnets/tasks/dualnets/configs/2_diracs.json'
out_path = './out/sweep_n_m_r_all_at_sphere_dim={d}/'.format(d=dim)

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
for m in [5, 10]:
    for n in [10, 20, 30, 40, 50, 60, 70, 80]:
        for r in radii_to_test:
            data['distrib2']['radius'] = r

            data['distrib1']['sample_size'] = m
            data['distrib2']['sample_size'] = n
            data['distrib1']['test_sample_size'] = m
            data['distrib2']['test_sample_size'] = n

            #write new config file
            with open(config_To_be_written, "w") as write_file:
                json.dump(data, write_file)

            #compute flat metric
            subprocess.call("python ./lnets/tasks/dualnets/mains/train_dual.py " + config_To_be_written, shell=True)

