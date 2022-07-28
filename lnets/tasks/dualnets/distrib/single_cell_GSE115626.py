import numpy as np
import csv
import pandas as pd
from sklearn.decomposition import PCA
from lnets.tasks.dualnets.distrib.base_distrib import BaseDistrib

class SingleCellClusters(BaseDistrib):
    '''A class which provides access to single cell transcriptomic data of various cell types
    within the GSE115626 sample. The highly variable genes are already known but still need
    to be pre-processed

    Attributes:
    cluster - the cluster to return samples from, i.e. 'NB', 'TAP', 'aNSC0', 'aNSC1', 'aNSC2', 'qNSC1', 'qNSC2'
    path - the file path containing the 'variable_genes_raw.csv' file of the UMI counts of the highly variable genes of GSE115626
    dim - the PCA dimension which the raw data should be reduced to. This should be the same in both distributions specified in the .json file
    '''
    def __init__(self, config):
        self.cluster = config.cluster
        self.path = config.path
        self.dim = config.dim
        self.samples = []

        self.__process_data__()

        print('Test here. Filepath=', self.path)

    def __PCA_reduction(self, centered_data, n_dimensions=5):
        pca = PCA(n_dimensions)
        principalComponents = pca.fit_transform(centered_data) 
        return principalComponents

    def __process_data__(self):
        '''
        This method performs the steps necessary to go from the UMI counts of the highly variable genes
        (= the content of the 'variable_genes_raw.csv' file) to the ready-to-return data samples.
        The following steps are necessary: 
        1) load whole data set as pd dataframe
        2)convert to np array, do preprocessing
        3)run PCA
        4)filter out relevant indices of cluster, fill sample list accordingly
        '''

        #1) load whole data set as pd dataframe
        data = pd.read_csv(self.path)

        # small mistakes during creation of variable_genes_raw, so need to correct those manually
        data.drop('sample', axis=1, inplace=True) #drop 'samples' column
        data.rename(columns = {'Unnamed: 0':'sample'}, inplace = True)# rename 'Unnamed: 0' as 'sample'

        # convert to np array
        data_np = data.to_numpy()[:,2:].astype(float) #get rid of labels in the first 2 columns

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
        principal_components = self.__PCA_reduction(data_centered, self.dim)

        #4) filter out relevant indices of cluster, fill sample list accordingly
        cell_types = ['NB', 'TAP', 'aNSC0', 'aNSC1', 'aNSC2', 'qNSC1', 'qNSC2'] #ignore 'OD', 'OPC' as they only have a small number4 of cells
        # This is hard-coded right now :(
        cell_dict = dict(zip(range(len(cell_types)), cell_types))
        cell_dict_by_name = {v: k for k, v in cell_dict.items()} # kinda stupid to create 2nd dictionary, but seems most comfortable approach

        #relevant_indices is array of arrays, with index specifiying kind of cluster according to cell_dict
        relevant_indices = np.array([data.index[data['cell_type']==cell_dict[key]].to_numpy() for key in cell_dict],dtype=object) #.keys() together with list comprehension requires the values to be indices
        
        relevant_samples = principal_components[relevant_indices[cell_dict_by_name[self.cluster]],:]
        self.samples = relevant_samples.tolist()

    def __call__(self, size):
        '''
        The method returning samples of the cluster specified in the config. Note that our
        samples are drawn uniformly from the actual data set such that we can 
        return arbitrarily many samples with finite data. 
        '''


        idx = np.random.randint(0, len(self.samples), size)
        #print('Laenge der Liste der idx: ', len(idx))
        
        #samples = np.array(self.samples)[idx]
        
        samples = np.random.permutation(np.array(self.samples))
        
        """
        if self.cluster == 'TAP':
            samples = np.loadtxt('data/samples1')
        elif self.cluster == 'NB':
            samples = np.loadtxt('data/samples2')
        """
        
        return samples