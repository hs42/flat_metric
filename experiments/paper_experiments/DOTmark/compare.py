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
from PIL import Image


import sys

__basedir__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir)
__filedir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(__basedir__, 'lnets', 'data'))


flat = True #if false, will use usual Wasserstein computation instead of flat metric

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

def load_and_process_img(path, size):
    img1 = Image.open(path).convert('L')
    img1 = img1.resize(size, resample=Image.NEAREST )
    img1 = np.array(img1)

    if len(np.unique(img1)) > 2:
        raise RuntimeError('Current img ist non-binary')
    data1 = []
    #Convert img into suitable data samples for empiric distributions
    for i in range(size[0]):
        for j in range(size[1]):
            if img1[i,j] == 255: #needs to be adjusted for gray scale images which are not binary
                data1.append(np.array([i / (size[0] - 1), j / (size[1] - 1)])) #support points in domain [0,1]^2
        
    data1 = np.array(data1)
    np.savetxt('mu', data1)
    return ('mu', np.shape(data1)[0])

def load_and_process_img_grayscale(path, size):
    img1 = Image.open(path).convert('L')
    img1 = img1.resize(size, resample=Image.NEAREST )
    img1 = np.array(img1)

    bins = np.linspace(0, 255, 10)
    repetitions = np.digitize(img1, bins)
   
    data1 = []
    #Convert img into suitable data samples for empiric distributions
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(repetitions[i,j]): 
                data1.append(np.array([i / (size[0] - 1), j / (size[1] - 1)])) #support points in domain [0,1]^2
        
    data1 = np.array(data1)
    np.savetxt('mu', data1)
    return ('mu', np.shape(data1)[0])


#read data
with open(path_to_default) as f:
   data = json.load(f)


#set certain parameters
data['model']['linear']['type'] = linear_layer_type
data['model']['name'] = model_name

data['distrib1']['dim'] = dim
data['distrib2']['dim'] = dim

data['cuda'] = use_cuda
data['logging']['save_best'] = save_best_model

Delta_pixel_at_O_mass = 1000
Delta_pixel_at_O_data = 'nu'
nu = np.zeros((Delta_pixel_at_O_mass, 2))
np.savetxt(Delta_pixel_at_O_data, nu)

img_list = ['picture32_10{i:02}.png'.format(i=i) for i in range (1,11)]
sizes = (32, 32)


for img in img_list:
    print('Compare compute distance to ', img)
    
    data['output_root'] = os.path.join(out_path, img[:-4])
    os.makedirs(os.path.join(out_path, img[:-4]))

    img_data_path, img_data_len = load_and_process_img_grayscale(os.path.join('Pictures', 'CauchyDensity', img), sizes)


    data['distrib1']['sample_size'] = img_data_len
    data['distrib2']['sample_size'] = Delta_pixel_at_O_mass


    data['distrib1']['path_train'] = os.path.join(__filedir__, img_data_path)
    data['distrib2']['path_train'] = os.path.join(__filedir__, Delta_pixel_at_O_data)
    
    data['optim']['epochs'] = 50000

    shutil.copy('mu', os.path.join(out_path, img[:-4], 'mu'))
    shutil.copy('nu', os.path.join(out_path, img[:-4], 'nu'))

    #write new config file
    with open(config_To_be_written, "w") as write_file:
        json.dump(data, write_file, indent=4)

    #compute flat metric between type1 and type2
    subprocess.call("python " + os.path.join(__basedir__, "lnets", "tasks", "dualnets", "mains","train_dual.py") + " " + config_To_be_written, shell=True)


#clean up
tempdir.cleanup()
sys.path.remove(os.path.join(__basedir__, 'lnets', 'data'))