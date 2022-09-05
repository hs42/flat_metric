from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
from custom_dataset import *
import json
import subprocess


def load_data(path):
    #1) load whole data set as pd dataframe
    data = pd.read_csv(path)

    # small mistakes during creation of variable_genes_raw, so need to correct those manually
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

    #2) do preprocessing
    '''
    from Ocima's paper:

    The sum of expression values for a cell is known as Library Size. Due to the way the scRNA-seq data is captured,
    there are some systematic differences between the library sizes across the cells. To compare the expression profiles between
    different cells, we perform the above normalization. This seems to be the easiest (and widely used) method to normalize for
    library size. More sophisticated methods also exist.
    '''
    lib_size = np.sum(data_np, axis = 1) # total expression value per cell, i.e. sum over genes

    tmp = np.repeat(lib_size, np.shape(data_np)[1]).reshape(np.shape(data_np)) # does the correct thing
    data_np = np.log(data_np / tmp * 1e4 + 1)

    # centering of data
    '''
    A normalization of the features is ususally required for PCA. Here, we scrap that step as the features 
    are by definition already comparable with each other (expression counts of genes) and only center the data.
    '''

    N = np.shape(data_np)[0] # total number of cells
    mu_g = np.sum(data_np, axis = 0) / N  

    data_centered = (data_np - mu_g) 

    #3) run PCA
    principal_components = __PCA_reduction(data_centered, dim_PCA)

    return principal_components

def save_PCA(outpath, PC, raw_data):
    n_dim = np.shape(PC)[1]
    col_names = ["PC{j}".format(j=i+1) for i in range(n_dim)]
    nice_principal_components = pd.DataFrame(PC, columns=col_names)
    nice_principal_components['cell_type'] = raw_data['cell_type']
    sign = np.sign(nice_principal_components[nice_principal_components['cell_type'] == 'Group1']['PC1'][0])
    nice_principal_components.loc[nice_principal_components['cell_type'] == 'Group1', 'PC1'] = nice_principal_components[nice_principal_components['cell_type'] == 'Group1']['PC1'] - sign*5
    
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
      
        np.savetxt(os.path.join(outpath, 'val_samples1'), s1_test)
        np.savetxt(os.path.join(outpath, 'val_samples2'), s2_test)

        

    else:
        s1_train, s2_train = s1.to_numpy(), s2.to_numpy()

    np.savetxt(os.path.join(outpath, 'train_samples1'), s1_train)
    np.savetxt(os.path.join(outpath, 'train_samples2'), s2_train)

    return (len(s1_train), len(s1_test) if val_fraction > 0 else 0, len(s2_train), len(s2_test) if val_fraction > 0 else 0)
   
"""
config values
"""
path_to_raw = "data/genes_raw_splatter_untersch_Anzahl.csv"
#path_to_raw = "data/variable_genes_raw.csv"
path_to_save_processed = "data/samples"

dim_PCA = 2
#cell_types = ['NB', 'TAP', 'aNSC0', 'aNSC1', 'aNSC2', 'qNSC1', 'qNSC2']
cell_types = ['Group1', 'Group2']#, 'Group3', 'Group4']

linear_layer_type = 'spectral_normal'

flat = True #if false, will use usual Wasserstein computation instead of flat metric

path_to_default =      './lnets/tasks/dualnets/configs/single_cell_data_revised.json'
config_To_be_written = './lnets/tasks/dualnets/configs/single_cell_comparison.json'
model_name = "dual_fc_flat" if flat else "dual_fc"
#out_path = "./out/val_test2_single_cell_comparison_flat" if flat else "./out/single_cell_comparison_Wasserstein"
out_path = "out/3_splatter_test_data_untersch_Anzahl"

"""
loading and preprocessing
"""
if not os.path.isdir(path_to_save_processed):
    os.makedirs(path_to_save_processed)

# if out_path already exists from a previous experiment
if os.path.isdir(out_path):
    raise RuntimeError('output directory already exists: ', out_path)
raw_data = load_data(path_to_raw)
principal_components = preprocess(raw_data, dim_PCA)
processed_data = save_PCA(path_to_save_processed, principal_components, raw_data)



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

results = pd.DataFrame(columns=['cell type 1', 'cell type 2', '{a} distance'.format(a='flat' if flat else 'Wassertein'), 'distance uncertainty'])

val_fraction = 0.0


#enter loop and sweep through different values for the radius
for type1 in ["Group1"]:#cell_types:
    for type2 in ["Group2"]:#cell_types:
        sample_sizes = generate_sample_datafiles(processed_data, path_to_save_processed, type1, type2, val_fraction)

        
        data['distrib1']['sample_size'] = sample_sizes[0]
        data['distrib2']['sample_size'] = sample_sizes[2]
        data['distrib1']['test_sample_size'] = sample_sizes[1]
        data['distrib2']['test_sample_size'] = sample_sizes[3]
        data['distrib1']['cluster'] = type1
        data['distrib2']['cluster'] = type2

        data['distrib1']['path_train'] = os.path.join(path_to_save_processed, 'train_samples1')
        data['distrib2']['path_train'] = os.path.join(path_to_save_processed, 'train_samples2')
        data['distrib1']['path_val'] = os.path.join(path_to_save_processed, 'val_samples1')
        data['distrib2']['path_val'] = os.path.join(path_to_save_processed, 'val_samples2')

        #write new config file
        with open(config_To_be_written, "w") as write_file:
            json.dump(data, write_file)

        #compute flat metric
        subprocess.call("python ./lnets/tasks/dualnets/mains/train_dual.py " + config_To_be_written, shell=True)


        """
        read logs and get results, store them in pandas dataframe
        """

        dirs = os.listdir(out_path)
        dirs.sort()
        path_recent = dirs[-1]

        distance_train = np.loadtxt(os.path.join(out_path, path_recent, 'logs/train_loss.log'), skiprows=1, delimiter=',')
        distance_estimate = -np.mean(distance_train[:-10:-1,1])
        distance_std = np.std(distance_train[:-10:-1,1])
        results.loc[len(results.index)]=[type1, type2, distance_estimate, distance_std]

results_table = results.pivot_table(index=['cell type 1'], columns=['cell type 2'], values='{a} distance'.format(a='flat' if flat else 'Wassertein'))
print(results_table)

#Saving to a pickled file gives a error when reading for some reason
#results.to_pickle(os.path.join(out_path, 'results_comparison_{a}'.format(a='flat' if flat else 'Wassertein')))
results.to_csv(os.path.join(out_path, 'results_comparison_{a}.csv'.format(a='flat' if flat else 'Wassertein')))
