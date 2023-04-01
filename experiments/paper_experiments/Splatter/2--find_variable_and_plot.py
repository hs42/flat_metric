"""
This script does some minor pre-processing. It reads in the UMI counts of the simualted cells ("counts_splatter"), and links
each cell to its corresponding group label found in "types_splatter". These are stored in a pandas dataframe object and saved to 
a csv file. If you wish, you can also create a t-SNE plot of the cell populations here.
"""

visualize = True

import pandas as pd
import numpy as np

#If you wish to create a t-SNE plot, need to load appropriate libraries
if visualize:
    from sklearn.manifold import TSNE
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA


"""
Read in data
"""

#read csv data
df = pd.read_csv("counts_splatter.csv")
df = df.transpose()

names = pd.read_csv("types_splatter.csv")
types_correct = list(names["Group"])

df.columns = list(df.iloc[0])
df = df.iloc[1:]

"""
Find highly variable genes
"""

means = df.mean().values
var = df.var().values
means[means == 0] = 0.001 #so that we don't divide by zero

indices = var / means > 1
df = df.loc[:, indices]
df.insert(0, "cell_type", types_correct)

"""
Save pandas df to be read in the compute_pairwise_distances script
"""

#create dummy columns; don't need to alter pre-processing steps from my other script
df.insert(0, "Unnamed: 0", "") 
df.insert(0, "sample", "") 

#df.insert(0, "dummy", np.arange(len(df)))
df.set_index("sample", inplace = True)

df.to_csv("genes_splatter.csv")

"""
Plot TSNE
"""
if visualize:
    """
    Pre-processing
    """

    data_np = df.to_numpy()[:,2:].astype(float) #get rid of labels in the first two columns
    lib_size = np.sum(data_np, axis = 1) # total expression value per cell, i.e. sum over genes

    tmp = np.repeat(lib_size, np.shape(data_np)[1]).reshape(np.shape(data_np)) # does the correct thing
    data_np = np.log(data_np / tmp * 1e4 + 1)

    N = np.shape(data_np)[0] # total number of cells
    mu_g = np.sum(data_np, axis = 0) / N  

    data_z_norm = (data_np - mu_g) # centering

    """
    PCA dimension reduction
    """

    def do_PCA(n_dimensions=5):
        pca = PCA(n_dimensions)
        principalComponents = pca.fit_transform(data_z_norm) # data_z_norm is already centered
        print(pca.explained_variance_ratio_)
        return principalComponents

    """
    Actually create t-SNE plot
    """

    PC_tsne = do_PCA(5)

    n_components = 2
    tsne = TSNE(n_components, perplexity=15)#, perplexity=45
    # Apparently, it is common to first do a PCA and then t-SNE
    red_data= tsne.fit_transform(PC_tsne) #PC_tsne has shape (n_samples, n_features= PCA features)


    """
    Adjust layout
    """
    sns.set(font_scale=1.2)
    red_data_df = pd.DataFrame({'tsne_1': red_data[:,0], 'tsne_2': red_data[:,1], 'label': df['cell_type'][:]})
    #red_data_df = pd.DataFrame({'PCA_1': PC_tsne[:,0], 'PCA_2': PC_tsne[:,1], 'label': df['cell_type'][:]})
    fig, ax = plt.subplots(1)
    colors1 = ['darkgoldenrod', 'cornflowerblue', 'lawngreen', 'yellow', 'lightpink']#, 'mediumblue', 'orange', 'peru', 'deeppink']
    colors2 = []
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=red_data_df, legend=False, ax=ax,s=30, palette=colors1)
    #sns.scatterplot(x='PCA_1', y='PCA_2', hue='label', data=red_data_df, legend='full', ax=ax,s=30, palette=colors1)
    lim_x = (-60, 80)
    lim_y = (-75, 65)
    #lim = (-10,10)
    ax.set_xlim(lim_x)
    ax.set_ylim(lim_y)
    ax.set_aspect('equal')
    #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    ax.set(xlabel='tsne 1')
    ax.set(ylabel='tsne 2')

    txts = []

    labels = np.unique(red_data_df['label'].values)
    print(labels)
    for i in labels:

        # Position of each label at median of data points.

        xtext = np.median(red_data_df[red_data_df['label'] == i]['tsne_1'].values) - 20
        ytext = np.median(red_data_df[red_data_df['label'] == i]['tsne_2'].values)

        if i == 'Group1':
            ytext -= 8

        txt = ax.text(xtext, ytext, str(i), fontsize=14,bbox={'facecolor':'white','alpha':0.2,'edgecolor':'none','pad':1, 'linewidth':10})


        txts.append(txt)

    fig.savefig('tSNE_splatter.png', format='PNG', dpi=300,  bbox_inches='tight')