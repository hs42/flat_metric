import json
import numpy as np
import subprocess
from munch import Munch
import os
from sklearn.model_selection import train_test_split

import sys
sys.path.append('.{s}lnets{s}tasks{s}dualnets{s}distrib'.format(s=os.sep))
import sum_of_Diracs


'''
quick / most important parameters
'''
dim = 2
val_fraction = 0.1
few_samples = True


if few_samples:
    sample_size_factor =  2**dim  
else:
    sample_size_factor = 6 * 2**dim  #account for need for more training data in higher dimensions


m_to_test = sample_size_factor*np.array([5, 10, 20], dtype=int)
n_to_test = sample_size_factor*np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=int)
l_fractions_to_test = np.arange(0, 1.1, 0.1)

groundtruth = np.zeros((len(m_to_test), len(n_to_test), len(l_fractions_to_test)))

mock_paras = Munch({'dim': dim, 'center_x' : [0.0], 'l' : 0, 'l_fraction': 0.0})


linear_layer_type = 'spectral_normal'

path_to_default =      '.{s}lnets{s}tasks{s}dualnets{s}configs{s}default_2_diracs_not_in_sphere_dl.json'.format(s=os.sep)
config_To_be_written = '.{s}lnets{s}tasks{s}dualnets{s}configs{s}2_diracs_not_in_sphere.json'.format(s=os.sep)
out_path = '.{s}out{s}Probe_Dirac_and_arbitrary_measure_with_dataloader{s}{n}_samples{s}dimension_{d}'.format(d=dim, s=os.sep, n=['many', 'few'][few_samples])

#read data
with open(path_to_default) as f:
   data = json.load(f)


#set certain parameters
data['model']['linear']['type'] = linear_layer_type

data['distrib1']['dim'] = dim 
data['distrib2']['dim'] = dim
data['output_root'] = out_path

#check if output path already exists and create new one
if os.path.exists(out_path):
    raise RuntimeError('Outpath dir already exists. Will not overwrite that!')
else:
    os.makedirs(out_path)

os.system("rm " + out_path + "{s}*samples*".format(s=os.sep)) #delete possible previous data sample files


#enter loop and sweep through different values for the radius
for m_i, m in enumerate(m_to_test):
    for n_i, n in enumerate(n_to_test):
        for l_i, l_f in enumerate(l_fractions_to_test):
            l = np.floor(l_f * n).astype(int)
            mock_paras.l = l
            mock_paras.l_fraction = l_f
            data_generator = sum_of_Diracs.Sum_of_Diracs_with_different_radii(mock_paras)
            groundtruth[m_i, n_i, l_i] = data_generator.get_groundtruth(m)
            s1 = np.zeros((m, dim))
            s2 = np.array(data_generator(n))
            if val_fraction > 0.0:
                """
                Also note that as we consider a uniform distribution, we expect train and val to differ highly with only so little data point
                """
                s1_train = s1
                s2_train = s2
                mock_paras.l = np.floor(l_f * n * val_fraction).astype(int)
                data_generator = sum_of_Diracs.Sum_of_Diracs_with_different_radii(mock_paras)
                s1_test = np.zeros((int(m*val_fraction), dim))
                s2_test =  np.array(data_generator(min(int(n*val_fraction), mock_paras.l+1)))

                groundtruth[m_i, n_i, l_i] = data_generator.get_groundtruth(m)
            
                np.savetxt(os.path.join(out_path, 'val_samples1'), s1_test)
                np.savetxt(os.path.join(out_path, 'val_samples2'), s2_test)

            else:
                s1_train, s2_train = s1, s2

            np.savetxt(os.path.join(out_path, 'train_samples1'), s1_train)
            np.savetxt(os.path.join(out_path, 'train_samples2'), s2_train)


            data['distrib1']['sample_size'] = int(m)#was int64 before, which json didn't like
            data['distrib2']['sample_size'] = int(n)#was int64 before, which json didn't like
            data['distrib1']['test_sample_size'] = int(m)#was int64 before, which json didn't like
            data['distrib2']['test_sample_size'] = int(n)#was int64 before, which json didn't like
            data['distrib2']['l'] = int(l) #was int64 before, which json didn't like
            data['distrib2']['l_fraction'] = l_f

            data['distrib1']['path_train'] = os.path.join(out_path, 'train_samples1')
            data['distrib2']['path_train'] = os.path.join(out_path, 'train_samples2')
            data['distrib1']['path_val'] = os.path.join(out_path, 'val_samples1')
            data['distrib2']['path_val'] = os.path.join(out_path, 'val_samples2')


            #write new config file
            with open(config_To_be_written, "w") as write_file:
                json.dump(data, write_file, indent=4)

            #compute flat metric
            
            subprocess.call("python .{s}lnets{s}tasks{s}dualnets{s}mains{s}train_dual.py ".format(s=os.sep) + config_To_be_written, shell=True)

np.save(os.path.join(out_path, 'groundtruth'), groundtruth)

os.remove(os.path.join(out_path, 'train_samples1'))
os.remove(os.path.join(out_path, 'train_samples2'))
os.remove(os.path.join(out_path, 'val_samples2'))
os.remove(os.path.join(out_path, 'val_samples1'))