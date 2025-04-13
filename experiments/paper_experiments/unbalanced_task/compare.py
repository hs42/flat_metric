from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import json
import subprocess
import tkinter as tk
from tkinter.filedialog import askdirectory
import tempfile
import shutil

import sys

__basedir__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir)
__filedir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(__basedir__, 'lnets', 'data'))


flat = False #if false, will use usual Wasserstein computation instead of flat metric

use_cuda = False #whether to use GPU or not
save_best_model = False #whether or not to store the best model for each training. These will be stored in the training output directory under 'checkpoints'
model_name = "dual_fc_flat" if flat else "dual_fc"

linear_layer_type = 'spectral_normal'

data_points_to_consider = 50 #how many epochs from the end of training to use to take the average over when computing the distance

dim = 2

"""
path configurations
"""
tempdir = tempfile.TemporaryDirectory() #create temporary dir where text files of the the currently celltypes under question are stored
path_to_save_processed = os.path.join(tempdir.name, 'samples')
if os.path.isdir(path_to_save_processed): #reset
    shutil.rmtree(path_to_save_processed)
os.makedirs(path_to_save_processed)


path_to_default =      os.path.join(__basedir__, 'lnets{s}tasks{s}dualnets{s}configs{s}default_datasets.json'.format(s=os.sep)) #default architecture for the net
config_To_be_written = os.path.join(path_to_save_processed, 'domain_adaptation_comparison.json') #adapt it such that the correct input is given in each loop

#ask user where the training results shall be strored
root = tk.Tk()
root.withdraw()
out_path_parent = askdirectory(title='Select empty folder for the output of this experiment', initialdir=os.path.join(__basedir__, 'out'))
# if out_path already exists from a previous experiment
if len(os.listdir(out_path_parent)) != 0:
    raise RuntimeError('output directory is not empty. Please choose another one', out_path_parent)
out_path = os.path.join(out_path_parent, 'training')
out_path_results = os.path.join(out_path_parent, 'results')

#and create sub-directories accordingly
os.makedirs(out_path)
os.makedirs(out_path_results)


"""
main loop
"""


#read data
with open(path_to_default) as f:
   data = json.load(f)


#set certain parameters
data['model']['linear']['type'] = linear_layer_type
data['model']['name'] = model_name

data['distrib1']['dim'] = dim
data['distrib2']['dim'] = dim

data['output_root'] = out_path
data['cuda'] = use_cuda
data['logging']['save_best'] = save_best_model

for s1 in ['A_before', 'B_before', 'C_before']:
    for s2 in ['A_after', 'B_after', 'C_after']:
        print('Compare now ', s1, 'and', s2)
        s1data = np.loadtxt(s1)
        s2data = np.loadtxt(s2)

        data['distrib1']['sample_size'] = len(s1data)
        data['distrib2']['sample_size'] = len(s2data)


        data['distrib1']['path_train'] = os.path.join(__filedir__, s1)
        data['distrib2']['path_train'] = os.path.join(__filedir__, s2)
        

        #write new config file
        with open(config_To_be_written, "w") as write_file:
            json.dump(data, write_file, indent=4)

        #compute flat metric between type1 and type2
        subprocess.call("python " + os.path.join(__basedir__, "lnets", "tasks", "dualnets", "mains","train_dual.py") + " " + config_To_be_written, shell=True)


#clean up
tempdir.cleanup()
sys.path.remove(os.path.join(__basedir__, 'lnets', 'data'))