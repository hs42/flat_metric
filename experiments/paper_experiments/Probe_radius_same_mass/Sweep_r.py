import json
import numpy as np
import subprocess
import os
import warnings
import tkinter as tk
from tkinter.filedialog import askdirectory
import tempfile
import sys


__basedir__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir)
__filedir__ = os.path.dirname(os.path.abspath(__file__))


"""
quick / most important parameters
"""
use_cuda = True #whether to use GPU or not
save_best_model = False #whether or not to store the best model for each training. These will be stored in the training output directory under 'checkpoints

dim = 5
few_samples = True

if few_samples:
    sample_size = 5 * 2**dim #account for need for more training data in higher dimensions
else:
    sample_size = 30 * 2**dim #account for need for more training data in higher dimensions

#radii_to_test = np.arange(radius_start, radius_stop, radius_step)
radii_to_test = [1,2]#list(np.arange(0.01, 3, 0.1)) + list(range(3,33,5)) #the radius which are to be tested in this script.

linear_layer_type = 'spectral_normal'


#quick check if we are able to handle the given amount of samples
if sample_size > 40000: #some threshold
    warnings.warn('Too many samples. Numpy might not be able to handle that. Will reduce samples to 40000') 
    sample_size = 40000

#ask user where the training results shall be strored
root = tk.Tk()
root.withdraw()
out_path_parent = askdirectory(title='Select empty folder for the output of this experiment', initialdir=os.path.join(__basedir__, 'out'))
# if out_path already exists from a previous experiment
if len(os.listdir(out_path_parent)) != 0:
    raise RuntimeError('output directory is not empty. Please choose another one', out_path_parent)
out_path = os.path.join(out_path_parent, 'training')


"""
path configurations
"""
tempdir = tempfile.TemporaryDirectory() #create temporary dir where text files of the the currently celltypes under question are stored
path_to_save_processed = os.path.join(tempdir.name, 'samples')
if os.path.isdir(path_to_save_processed): #reset
    shutil.rmtree(path_to_save_processed)
os.makedirs(path_to_save_processed)

path_to_default =      os.path.join(__basedir__, 'lnets{s}tasks{s}dualnets{s}configs{s}default_2_diracs.json'.format(s=os.sep))
config_To_be_written = os.path.join(tempdir.name, '2_diracs.json')
out_path = os.path.join(out_path, '{n}_samples{s}dimension_{d}'.format(d=dim, s=os.sep, n=['many', 'few'][few_samples]))

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
data['cuda'] = use_cuda
data['logging']['save_best'] = save_best_model

"""
enter loop and sweep through different values for the radius
"""
for r in radii_to_test:
    data['distrib2']['radius'] = r

    #write new config file
    with open(config_To_be_written, "w") as write_file:
        json.dump(data, write_file, indent=4)

    #compute flat metric
    subprocess.call("python " + os.path.join(__basedir__, "lnets", "tasks", "dualnets", "mains","train_dual.py") + " " + config_To_be_written, shell=True)


#clean up
tempdir.cleanup()