import json
import numpy as np
import subprocess
from munch import Munch

import sys
sys.path.append('./lnets/tasks/dualnets/distrib')
import sum_of_Diracs

'''
quick / most important parameters
'''
dim = 2

samples_To_be_written = './lnets/tasks/dualnets/distrib/sample_points_sum_of_Diracs'

#m_to_test = [5, 10]
#n_to_test = [10, 20, 30, 40, 50, 60, 70, 80]
m_to_test = [20, 40]
n_to_test = [100, 200, 300, 400, 500, 600, 700, 800]
l_fractions_to_test = np.arange(0, 1.1, 0.1)

groundtruth = np.zeros((len(m_to_test), len(n_to_test), len(l_fractions_to_test)))

mock_paras = Munch({'dim': dim, 'center_x' : [0.0], 'l' : 0, 'l_fraction': 0.0})


linear_layer_type = 'spectral_normal'

path_to_default =      './lnets/tasks/dualnets/configs/default_2_diracs_not_in_sphere.json'
config_To_be_written = './lnets/tasks/dualnets/configs/2_diracs_not_in_sphere.json'
out_path = './out/few_hp_large_ds_sweep_m_n_l_dim={d}_with_intermediate_check_of_penalty_Adam_decaying_lr_penalty_002_long/'.format(d=dim)

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
for m_i, m in enumerate(m_to_test):
    for n_i, n in enumerate(n_to_test):
        for l_i, l_f in enumerate(l_fractions_to_test):
            l = np.floor(l_f * n).astype(int)
            mock_paras.l = l
            mock_paras.l_fraction = l_f
            data_generator = sum_of_Diracs.Sum_of_Diracs_with_different_radii(mock_paras)
            samples = np.array(data_generator(n))
            groundtruth[m_i, n_i, l_i] = data_generator.get_groundtruth(m)


            #write samples to file
            np.savetxt(samples_To_be_written, samples)

            data['distrib1']['sample_size'] = m
            data['distrib2']['sample_size'] = n
            data['distrib1']['test_sample_size'] = m
            data['distrib2']['test_sample_size'] = n
            data['distrib2']['l'] = int(l) #was int64 before, which json didn't like
            data['distrib2']['l_fraction'] = l_f


            #write new config file
            with open(config_To_be_written, "w") as write_file:
                json.dump(data, write_file)

            #compute flat metric
            subprocess.call("python ./lnets/tasks/dualnets/mains/train_dual.py " + config_To_be_written, shell=True)


np.save(out_path+'groundtruth', groundtruth)
