"""
In this script, we train various nets, each one on a pair of celltypes given by the Splatter simulation.
This training on the individual datasets is comparable to that seen in the tutorial_datasets folder.

To mimick real life data and its treatment. the dataset is pre-processed at first. Particularly, we normalize the
gene expressions by the activity of each cell, center the data, and to a PCA (to usually 5 dimensions). Each of the
5 simulated groups from Splatter is thus defined by these 5 features.
In a loop, the pairwise distances are then computed and stored in a pandas df object.

The results are stored in the output directory (will be created)

"""


from sklearn.decomposition import PCA
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

sys.path.append(os.path.join(__basedir__, 'lnets', 'tasks', 'dualnets', 'mains'))

from custom_dataset import *

def load_data(path):
    #1) load whole data set as pd dataframe
    data = pd.read_csv(path)

    # small mistakes in labelling during creation of variable_genes_raw, so need to correct those manually
    data.drop('sample', axis=1, inplace=True) #drop 'samples' column
    data.rename(columns = {'Unnamed: 0':'sample'}, inplace = True)# rename 'Unnamed: 0' as 'sample'

    return data

def __PCA_reduction(centered_data, n_dimensions=5):
    pca = PCA(n_dimensions)
    principalComponents = pca.fit_transform(centered_data) 
    return principalComponents


def preprocess(data_raw, dim_PCA=5):
    # convert to np array
    data_np = data_raw.to_numpy()[:,2:].astype(float) #get rid of labels in the first 2 columns

    lib_size = np.sum(data_np, axis = 1) # total expression value per cell, i.e. sum over genes. This is known as the Library size and differs from cell to cell in actual experiments

    tmp = np.repeat(lib_size, np.shape(data_np)[1]).reshape(np.shape(data_np)) # does the correct thing
    data_np = np.log(data_np / tmp * 1e4 + 1)

    N = np.shape(data_np)[0] # total number of cells
    mu_g = np.sum(data_np, axis = 0) / N  

    #center the data
    data_centered = (data_np - mu_g) 

    #3) run PCA
    principal_components = __PCA_reduction(data_centered, dim_PCA)

    return principal_components

def save_PCA(outpath, PC, raw_data):
    n_dim = np.shape(PC)[1]
    col_names = ["PC{j}".format(j=i+1) for i in range(n_dim)]
    nice_principal_components = pd.DataFrame(PC, columns=col_names)
    nice_principal_components['cell_type'] = raw_data['cell_type']
    
    nice_principal_components.to_pickle(os.path.join(outpath, "processed_data_PCA_dim={d}.pkl".format(d=n_dim)))
    return nice_principal_components


def generate_sample_datafiles(processed_data, outpath, type1, type2, val_fraction=0.0):
   
    threshold_too_few_samples = 20

    s1 = processed_data[processed_data['cell_type']==type1]
    s2 = processed_data[processed_data['cell_type']==type2]

    s1=s1.drop('cell_type', axis=1) #get rid of 'cell_type' as its not needed anymore
    s2=s2.drop('cell_type', axis=1) #get rid of 'cell_type' as its not needed anymore


    if len(s1) <= threshold_too_few_samples:
        raise RuntimeError('Too few samples for cell cluster', type1, '. Does it exist? Will abort now')
    if len(s2) <= threshold_too_few_samples:
        raise RuntimeError('Too few samples for cell cluster', type2, '. Does it exist? Will abort now')

    if val_fraction > 0.0:
        s1_train, s1_test = train_test_split(s1.to_numpy(), test_size=val_fraction)
        s2_train, s2_test = train_test_split(s2.to_numpy(), test_size=val_fraction)

        #if we test, split non-training set into validation set (run after each epoch) and a test set (run after training) in equal parts
        s1_val, s1_test = train_test_split(s1_test, test_size=0.5)
        s2_val, s2_test = train_test_split(s2_test, test_size=0.5)
      
        np.savetxt(os.path.join(outpath, 'val_samples1'), s1_val)
        np.savetxt(os.path.join(outpath, 'val_samples2'), s2_val)
        np.savetxt(os.path.join(outpath, 'test_samples1'), s1_test)
        np.savetxt(os.path.join(outpath, 'test_samples2'), s2_test)

        

    else:
        s1_train, s2_train = s1.to_numpy(), s2.to_numpy()

    np.savetxt(os.path.join(outpath, 'train_samples1'), s1_train)
    np.savetxt(os.path.join(outpath, 'train_samples2'), s2_train)

    return (len(s1_train), len(s1_val) if val_fraction > 0 else 0, len(s1_test) if val_fraction > 0 else 0, 
        len(s2_train), len(s2_val) if val_fraction > 0 else 0, len(s2_test) if val_fraction > 0 else 0)
   
"""
config values
"""

flat = True #if false, will use usual Wasserstein computation instead of flat metric

use_cuda = True #whether to use GPU or not
save_best_model = False #whether or not to store the best model for each training. These will be stored in the training output directory under 'checkpoints'
model_name = "dual_fc_flat" if flat else "dual_fc"

dim_PCA = 5
cell_types = ['Group1', 'Group2' 'Group3', 'Group4', 'Group5'] #the groups which are to be analyzed. Need to match the names specified in the genes_splatter.csv file

linear_layer_type = 'spectral_normal'

data_points_to_consider = 50 #how many epochs from the end of training to use to take the average over when computing the distance


"""
path configurations
"""
tempdir = tempfile.TemporaryDirectory() #create temporary dir where text files of the the currently celltypes under question are stored
path_to_save_processed = os.path.join(tempdir.name, 'samples')
if os.path.isdir(path_to_save_processed): #reset
    shutil.rmtree(path_to_save_processed)
os.makedirs(path_to_save_processed)


path_to_raw = os.path.join(__filedir__, 'genes_splatter.csv') #read in gene UMI counts = INPUT


path_to_default =      os.path.join(__basedir__, 'lnets{s}tasks{s}dualnets{s}configs{s}single_cell_data_revised.json'.format(s=os.sep)) #default architecture for the net
config_To_be_written = os.path.join(path_to_save_processed, 'single_cell_comparison.json') #adapt it such that the correct input is given in each loop

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
loading and preprocessing
"""
raw_data = load_data(path_to_raw)
principal_components = preprocess(raw_data, dim_PCA)
processed_data = save_PCA(out_path_results, principal_components, raw_data)


"""
main loop
"""


#read data
with open(path_to_default) as f:
   data = json.load(f)


#set certain parameters
data['model']['linear']['type'] = linear_layer_type
data['model']['name'] = model_name

data['distrib1']['dim'] = dim_PCA 
data['distrib2']['dim'] = dim_PCA

data['output_root'] = out_path
data['cuda'] = use_cuda
data['logging']['save_best'] = save_best_model

results = pd.DataFrame(columns=['cell type 1', 'cell type 2', '{a} distance'.format(a='flat' if flat else 'Wassertein'), 'distance uncertainty'])

val_fraction = 0.1


#enter loop and sweep through different values for the radius
for type1 in cell_types:
    for type2 in cell_types:
        sample_sizes = generate_sample_datafiles(processed_data, path_to_save_processed, type1, type2, val_fraction)

        
        data['distrib1']['sample_size'] = sample_sizes[0]
        data['distrib2']['sample_size'] = sample_sizes[3]
        data['distrib1']['test_sample_size'] = sample_sizes[1]
        data['distrib2']['test_sample_size'] = sample_sizes[4]
        data['distrib1']['cluster'] = type1
        data['distrib2']['cluster'] = type2

        data['distrib1']['path_train'] = os.path.join(path_to_save_processed, 'train_samples1')
        data['distrib2']['path_train'] = os.path.join(path_to_save_processed, 'train_samples2')
        data['distrib1']['path_val'] = os.path.join(path_to_save_processed, 'val_samples1')
        data['distrib2']['path_val'] = os.path.join(path_to_save_processed, 'val_samples2')
        data['distrib1']['path_test'] = os.path.join(path_to_save_processed, 'test_samples1')
        data['distrib2']['path_test'] = os.path.join(path_to_save_processed, 'test_samples2')

        #write new config file
        with open(config_To_be_written, "w") as write_file:
            json.dump(data, write_file, indent=4)

        #compute flat metric between type1 and type2
        subprocess.call("python " + os.path.join(__basedir__, "lnets", "tasks", "dualnets", "mains","train_dual.py") + " " + config_To_be_written, shell=True)


        """
        Now that we have calculated the flat metric, get its value and store it in a pandas df
        read logs and get results, store them in pandas dataframe
        """

        dirs = os.listdir(out_path)
        dirs.sort()
        path_recent = dirs[-1]


        """
        quick fix as train_loss_W yields NaNs in the Wasserstein case (small bug)
        """
        if flat:
            input_file = 'logs{s}train_loss_W.log'.format(s=os.sep)
        else:
            input_file = 'logs{s}train_loss.log'.format(s=os.sep)

        distance_train = np.loadtxt(os.path.join(out_path, path_recent, input_file), skiprows=1, delimiter=',')
        distance_estimate = -np.mean(distance_train[:-data_points_to_consider:-1,1])
        distance_std = np.std(distance_train[:-data_points_to_consider:-1,1]) / np.sqrt(data_points_to_consider)
        results.loc[len(results.index)]=[type1, type2, distance_estimate, distance_std]


"""
transform results to easily-readable pivot table
"""
results_table = results.pivot_table(index=['cell type 1'], columns=['cell type 2'], values='{a} distance'.format(a='flat' if flat else 'Wassertein'))
print(results_table)

#Saving to a pickled file gives a error when reading for some reason
results.to_csv(os.path.join(out_path_results, 'results_comparison_{a}.csv'.format(a='flat' if flat else 'Wassertein')))

#clean up
tempdir.cleanup()