"""
This is the script to reproduce the "Experiments - Testing unequal masses and dropping the assumptions on the support" section of the paper.
Computes the flat distance between mu sitting at zero (mu=sample_size * m * delta(0)) and nu, which describes an arbitary empiric measure of mass
sample_size * n. The masses m and n are given in m_to_test and n_to_test, respectively. Specifically, nu is constructed in the following way:
Firstly, sample_size * n data points are sampled randomly on a sphere. Afterwards, each data point is radially scaled with an independet factor between
0 and 200 (specified in the invoked script lnets/tasks/dualnets/distrib/sum_of_Diracs.py). In doing so, it is ensured that a fraction of l_f of these data
points fall within a radius of no more than 2 to the origin, where mu sits. This is to test the different modi of transport versus creation/deletion of 
the unbalanced optimal transport problem. l_f is varied according to the values specified in l_fractions_to_test.

As for each new parameter set n, m, l_f new data points are sampled, the groundtruth will also differ from run to run (specifically the according to the
rounded value of l_f * n) - thus we need to compute it given the
speific data at hand for this run. The computed ground truth will be stored in a np array, which will be written to a file after all runs took place.
"""

import json
import numpy as np
import subprocess
from munch import Munch
import os
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter.filedialog import askdirectory
import tempfile
import sys
import shutil

__basedir__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir)
__filedir__ = os.path.dirname(os.path.abspath(__file__))



sys.path.append(os.path.join(__basedir__, 'lnets{s}tasks{s}dualnets{s}distrib'.format(s=os.sep)))
import sum_of_Diracs


'''
quick / most important parameters
'''
use_cuda = True #whether to use GPU or not
save_best_model = False #whether or not to store the best model for each training. These will be stored in the training output directory under 'checkpoints

dim = 2 #the dimension of the sphere on which the points should be sampled

#the fraction of data points which are used for validation and not training in each run. Note that the validation results do not get stored, 
#but if the save_best_model=True flag is set, the best model according to the validation loss will be saved in each run
val_fraction = 0.1 

few_samples = True


if few_samples:
    sample_size_factor =  2**dim  
else:
    sample_size_factor = 6 * 2**dim  #account for need for more training data in higher dimensions


m_to_test = sample_size_factor*np.array([5, 10, 20], dtype=int)
n_to_test = sample_size_factor*np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=int)
l_fractions_to_test = np.arange(0, 1.1, 0.1)


linear_layer_type = 'spectral_normal'

#ask user where the training results shall be strored
root = tk.Tk()
root.withdraw()
out_path_parent = askdirectory(title='Select empty folder for the output of this experiment', initialdir=os.path.join(__basedir__, 'out'))
# if out_path already exists from a previous experiment
if len(os.listdir(out_path_parent)) != 0:
    raise RuntimeError('output directory is not empty. Please choose another one', out_path_parent)
out_path = os.path.join(out_path_parent, 'training{s}{n}_samples{s}dimension_{d}'.format(d=dim, s=os.sep, n=['many', 'few'][few_samples]))
out_results = os.path.join(out_path_parent, 'results')
os.makedirs(out_results)


"""
Misc initialization and path configuratuin
"""
groundtruth = np.zeros((len(m_to_test), len(n_to_test), len(l_fractions_to_test)))

mock_paras = Munch({'dim': dim, 'center_x' : [0.0], 'l' : 0, 'l_fraction': 0.0})

path_to_default =      os.path.join(__basedir__, 'lnets{s}tasks{s}dualnets{s}configs{s}default_datasets.json'.format(s=os.sep))

tempdir = tempfile.TemporaryDirectory() #create temporary dir where text files of the the currently celltypes under question are stored
path_to_save_processed = tempdir.name
if os.path.isdir(path_to_save_processed): #reset
    shutil.rmtree(path_to_save_processed)
os.makedirs(path_to_save_processed)
config_To_be_written = os.path.join(path_to_save_processed, '2_diracs_not_in_sphere.json')


"""
Read default configuration and adapt parameters
"""
#read data
with open(path_to_default) as f:
   data = json.load(f)

#set certain parameters
data['model']['linear']['type'] = linear_layer_type

data['distrib1']['dim'] = dim 
data['distrib2']['dim'] = dim
data['output_root'] = out_path
data['cuda'] = use_cuda
data['logging']['save_best'] = save_best_model


"""
enter loop and sweep through different values for the radius, the masses and the fractions
"""
for m_i, m in enumerate(m_to_test):
    for n_i, n in enumerate(n_to_test):
        for l_i, l_f in enumerate(l_fractions_to_test):
            l = np.floor(l_f * n).astype(int) #convert desired fraction into amount of data points, which should reside within a radius of 2
            mock_paras.l = l
            mock_paras.l_fraction = l_f
            data_generator = sum_of_Diracs.Sum_of_Diracs_with_different_radii(mock_paras)
            groundtruth[m_i, n_i, l_i] = data_generator.get_groundtruth(m)
            s1 = np.zeros((m, dim)) #samples of mu: just a dim-dimensional dirac-delta in the origin
            s2 = np.array(data_generator(n)) #samples of nu are generated by invoking data_generator of the Sum_of_Diracs_with_different_radii class
            if val_fraction > 0.0:
                """
                Also note that as we consider a uniform distribution, we expect train and val to differ highly with only so little data point
                """
                s1_train = s1 #If we want to validate, we want to generate additional data points. Otherwise the displayed masses m, n are not what the net was trained on
                s2_train = s2
                mock_paras.l = np.floor(l_f * n * val_fraction).astype(int)
                data_generator = sum_of_Diracs.Sum_of_Diracs_with_different_radii(mock_paras)

                #generate new validation data here
                s1_test = np.zeros((int(m*val_fraction), dim))
                s2_test =  np.array(data_generator(min(int(n*val_fraction), mock_paras.l+1)))

                #compute the ground truth
                groundtruth[m_i, n_i, l_i] = data_generator.get_groundtruth(m)
            
                np.savetxt(os.path.join(path_to_save_processed, 'val_samples1'), s1_test)
                np.savetxt(os.path.join(path_to_save_processed, 'val_samples2'), s2_test)

            else:
                s1_train, s2_train = s1, s2

            np.savetxt(os.path.join(path_to_save_processed, 'train_samples1'), s1_train)
            np.savetxt(os.path.join(path_to_save_processed, 'train_samples2'), s2_train)


            data['distrib1']['sample_size'] = int(m)#was int64 before, which json didn't like
            data['distrib2']['sample_size'] = int(n)#was int64 before, which json didn't like
            data['distrib1']['test_sample_size'] = int(m)#was int64 before, which json didn't like
            data['distrib2']['test_sample_size'] = int(n)#was int64 before, which json didn't like
            data['distrib2']['l'] = int(l) #was int64 before, which json didn't like
            data['distrib2']['l_fraction'] = l_f

            #specify where the data files for this run lay
            data['distrib1']['path_train'] = os.path.join(path_to_save_processed, 'train_samples1')
            data['distrib2']['path_train'] = os.path.join(path_to_save_processed, 'train_samples2')
            data['distrib1']['path_val'] = os.path.join(path_to_save_processed, 'val_samples1')
            data['distrib2']['path_val'] = os.path.join(path_to_save_processed, 'val_samples2')


            #write new config file
            with open(config_To_be_written, "w") as write_file:
                json.dump(data, write_file, indent=4)

            #compute flat metric
            
            subprocess.call("python " + os.path.join(__basedir__, "lnets", "tasks", "dualnets", "mains","train_dual.py") + " " + config_To_be_written, shell=True)

#save ground truth. It will be read in by the visualizing script
np.save(os.path.join(out_results, 'groundtruth'), groundtruth)


#cleanup
tempdir.cleanup()
sys.path.remove(os.path.join(__basedir__, 'lnets{s}tasks{s}dualnets{s}distrib'.format(s=os.sep)))

