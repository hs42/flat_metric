"""
This is the script to reproduce the Experiments - Architectural hyperparameters section of the paper.
Compute the flat distance between mu of mass m sitting at zero (mu=m * delta(0)) and nu of mass n, which describes a 
sphere of radius r (nu=n * delta(radius r)). R is changed to the values specified in radii_to_test, but m and n are fixed.
We considered (arbitrarily) the case of m=sample_size_factor*10 and n=sample_size_factor*50

In the main loop, different architectural parameters are altered
"""

import json
import numpy as np
import subprocess
import os
import tempfile
import tkinter as tk
from tkinter.filedialog import askdirectory
import shutil

__basedir__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir)
__filedir__ = os.path.dirname(os.path.abspath(__file__))

'''
quick / most important parameters
'''
dim = 4
few_samples = True


use_cuda = False #whether to use GPU or not
save_best_model = False #whether or not to store the best model for each training. These will be stored in the training output directory under 'checkpoints


if few_samples:
    sample_size_factor =  2**dim 
else:
    sample_size_factor = 6 * 2**dim #account for need for more training data in higher dimensions


radii_to_test = list(np.arange(0.01, 3, 0.3)) + list(range(3,13,1)) #list concatenation


#ask user where the training results shall be strored
root = tk.Tk()
root.withdraw()
out_path_parent = askdirectory(title='Select empty folder for the output of this experiment', initialdir=os.path.join(__basedir__, 'out'))
# if out_path already exists from a previous experiment
if len(os.listdir(out_path_parent)) != 0:
    raise RuntimeError('output directory is not empty. Please choose another one', out_path_parent)
out_path = out_path_parent


"""
path configurations
"""
tempdir = tempfile.TemporaryDirectory() #create temporary dir where text files of the the currently celltypes under question are stored
path_to_save_processed = tempdir.name
if os.path.isdir(path_to_save_processed): #reset
    shutil.rmtree(path_to_save_processed)
os.makedirs(path_to_save_processed)

path_to_default =      os.path.join(__basedir__, 'lnets{s}tasks{s}dualnets{s}configs{s}default_2_diracs.json'.format(s=os.sep))
config_To_be_written = os.path.join(tempdir.name, '2_diracs.json')


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


"""
Set/Initialize parameters
"""

m = int(sample_size_factor*10) #chosen somewhat arbitrarily
n = int(sample_size_factor*50) #chosen somewhat arbitrarily

data['distrib1']['dim'] = dim 
data['distrib2']['dim'] = dim

data['distrib1']['sample_size'] = int(m)
data['distrib2']['sample_size'] = int(n)
data['distrib1']['test_sample_size'] = max(1,int(0.1*m))
data['distrib2']['test_sample_size'] = max(1,int(0.1*n))

data['cuda'] = use_cuda

linear_layer_type = 'spectral_normal'
layers = [] #the hidden layers. List of number of respective neurons
activation = 'group_sort'
groupings = [] #the number of neurons in each hidden layer used for calculating the GroupSort activation function. 
#E.g. groupings = [2, 4] would mean that in the first layer groups of two neurons each will be formed, and the neurons in each one will be compared and sorted,
#whereas it would be groups of 4 neurons for the second hidden layer.  

"""
#enter loop: modify architecutral parameters in each one. For each run, sweep through different values for the radius to observe which effects the hyperparameters have
"""
for i in range(6):

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
    elif i == 3:
        #make layers bigger
        linear_layer_type = 'spectral_normal'
        layers = [512, 512, 1]
        groupings = [8, 8, 1]
    elif i == 4:
        #make net deeper, add more training time, switch to Bjorck
        linear_layer_type = 'bjorck'
        layers = [128, 128, 128, 128, 128, 1]
        groupings = [2, 2, 2, 2, 2, 1]
    elif i == 5:
        #make layers bigger, switch to Bjorck
        linear_layer_type = 'bjorck'
        layers = [512, 512, 1]
        groupings = [8, 8, 1]


    data['model']['linear']['type'] = linear_layer_type
    data['model']['layers'] = layers
    data['model']['groupings'] = groupings
    data['model']['activation'] = activation


    out_path2 = os.path.join(out_path, 'study_case_'+str(i))
    data['output_root'] = out_path2

    for r in radii_to_test:
        data['distrib2']['radius'] = r

        #write new config file
        with open(config_To_be_written, "w") as write_file:
            json.dump(data, write_file, indent=4)

        #compute flat metric
        subprocess.call("python " + os.path.join(__basedir__, "lnets", "tasks", "dualnets", "mains","train_dual.py") + " " + config_To_be_written, shell=True)

#clean up
tempdir.cleanup()
