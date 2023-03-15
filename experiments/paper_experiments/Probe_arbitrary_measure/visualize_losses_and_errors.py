"""
This script will create plots akin to figure 6. It also creates plots like figure 3, this time however for the "Probe against arbitrary measure" experiment.
"""

import numpy as np 
import numpy.ma as ma
from matplotlib import pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tkinter as tk
from tkinter.filedialog import askdirectory, asksaveasfile
import os
import json

__basedir__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir)
__filedir__ = os.path.dirname(os.path.abspath(__file__))

#important variables
data_points_to_consider = 50 #the number of last epochs to consider in each experiment (=training of a nn) to infer the values by taking the mean
plot_uncertainties = False #tweak this parameter to either plot the data itself (False) or its standard deviations (True)



plot_uncertainties_par = {False: 0, True: 1}[plot_uncertainties]#convert boolean into an index which is specifically 0 or 1 -> can use that index in np arrays

def init():
    root = tk.Tk()
    root.withdraw()
    path = askdirectory(title='Select parent folder containing the out files of a \'Probe against arbitrary measure\'-experiment, \
        where the subfolders for many and few datapoints are stored', initialdir=os.path.join(__basedir__, 'out'))

    tmp = os.listdir(path)

    few_samples = False

    #check if necessary directories exist and ask user which case (few/many and dimension) they wish to plot 
    if 'many_samples' in tmp and 'few_samples' in tmp:
        tmp2 = input('Found data for many and few samples in this directory. Which one do you want to plot [few, many]? ')
        if tmp2 == 'few':
            few_samples = True
        elif tmp2 == 'many':
            few_samples = False
        else:
            raise ValueError('Didnt get expected keyword ’few’ or ’many’')
    elif 'many_samples' in tmp  and len(tmp) == 1:
        few_samples = False
    elif 'few_samples' in tmp  and len(tmp) == 1:
        few_samples = True

    else:
        raise RuntimeError('The specified path does not contain valid subdirectories \’few_samples\’ or \’many_samples\’')

    path2 = {True: 'few_samples', False: 'many_samples'}[few_samples]

    dim = 0
    tmp = os.listdir(os.path.join(path, path2))
    if len(tmp) == 0:
        raise RuntimeError('There is no data in ' + os.path.join(path, path2))
    elif len(tmp) == 1:
        path3 = tmp[0]
        dim = tmp[0][10:]
    else:
        print('Found the following dimensions: ')
        for d in tmp:
            print(d[10:], end=' ')
        dim = input('Which one do you want to analyse? ')
        if not float(dim)%1 == 0:
            raise ValueError('The provided dimension is not an integer')
        path3 = 'dimension_' + dim
        if path3 not in tmp:
            raise ValueError('The given dimension does not exist: found no directory called ' + path3)

    return (os.path.join(path, path2, path3), int(dim))

def load_data(path, dim_by_user):
    dirs = os.listdir(path)
    dirs.sort()
    number_of_dirs = int(len(dirs))

    l_fraction_values = np.zeros(number_of_dirs)
    m_values = np.zeros(number_of_dirs, dtype=int)
    n_values = np.zeros(number_of_dirs, dtype=int)

    distance_estimates = np.zeros((number_of_dirs, 2)) #dir_idx,[mean, std]
    actual_penalties = np.zeros((number_of_dirs, 2))   #dir_idx,[mean, std]

    print('Loading data ...')


    for i, dir in enumerate(dirs):
        with open(os.path.join(path, dir, 'logs{s}config.json'.format(s=os.sep))) as f:
            data = json.load(f)

        #r = abs(data['distrib2']['radius'] - data['distrib1']['radius'])
        l_f = data['distrib2']['l_fraction']
        n = data['distrib2']['sample_size']
        m = data['distrib1']['sample_size']

        dim = data['distrib2']['dim']
        if dim != dim_by_user:
            raise RuntimeError('The dimensions found do not match with the dimension label of the directory')
        linear_type = data['model']['linear']['type']

        #read in data
        distance_train = np.loadtxt(os.path.join(path, dir, 'logs{s}train_loss_W.log'.format(s=os.sep)), skiprows=1, delimiter=',')
        loss_flat = np.loadtxt(os.path.join(path, dir, 'logs{s}train_loss_flat.log'.format(s=os.sep)), skiprows=1, delimiter=',')
        lambda_penalties = np.loadtxt(os.path.join(path, dir, 'logs{s}lambda.log'.format(s=os.sep)), skiprows=1)


        #store important data
        distance_estimates[i, 0] = -np.mean(distance_train[:-data_points_to_consider:-1,1]) #mean of last data_points_to_consider entries
        distance_estimates[i ,1] = np.std(distance_train[:-data_points_to_consider:-1,1]) / np.sqrt(data_points_to_consider) #error of the mean of last data_points_to_consider entries

        actual_penalties[i, 0] = np.mean(lambda_penalties[:-data_points_to_consider:-1,1]) #mean of last 10 entries of actual_penalty
        actual_penalties[i, 1] = np.std(lambda_penalties[:-data_points_to_consider:-1,1]) / np.sqrt(data_points_to_consider) #error of the mean of last data_points_to_consider entries

        l_fraction_values[i] = l_f
        m_values[i] = int(m)
        n_values[i] = int(n)

    return (distance_estimates, actual_penalties, l_fraction_values, m_values, n_values)

def rearrange_data_to_make_it_plottable(path, distance_estimates, actual_penalties, l_fraction_values, m_values, n_values):
    unique_m = np.unique(m_values)
    unique_n = np.unique(n_values)
    unique_l_f = np.unique(l_fraction_values)

    distance_estimates_plot = np.zeros((len(unique_m), len(unique_n), len(unique_l_f), 2))
    penalties_plot = np.zeros((len(unique_m), len(unique_n), len(unique_l_f), 2))
    relative_errors_plot = np.zeros((len(unique_m), len(unique_n), len(unique_l_f), 2))

    relative_errors_ratios_plot = np.zeros((len(unique_m) * len(unique_n), len(unique_l_f)))
    ratio_n_to_m = np.zeros(len(unique_m) * len(unique_n))

    groundtruth = np.load(os.path.join(path, os.pardir, os.pardir, os.pardir, 'results', 'groundtruth.npy'))


    print('Rearranging data ...')

    #most straightforward to iterate over parameter space
    for i, m in enumerate(unique_m):
        for j, n in enumerate(unique_n):
            for k, l_f in enumerate(unique_l_f):
                mask = np.logical_and(np.logical_and(m_values == m, n_values == n), l_fraction_values == l_f)

                distance_estimates_plot[i, j, k, :] = distance_estimates[mask, :]
                penalties_plot[i, j, k, :] = actual_penalties[mask, :]
                
                relative_errors_plot[i, j, k, 0] = distance_estimates_plot[i, j, k, 0] / groundtruth[i, j, k]  - 1 
                relative_errors_plot[i, j, k, 1] = distance_estimates_plot[i, j, k, 1] / groundtruth[i, j, k]

                relative_errors_ratios_plot[i*len(unique_n)+j, k] = relative_errors_plot[i, j, k, plot_uncertainties_par]
                ratio_n_to_m[i*len(unique_n)+j] = n/m

    ordering = np.argsort(ratio_n_to_m)

    return (distance_estimates_plot, relative_errors_plot, penalties_plot, relative_errors_ratios_plot[ordering, :], ratio_n_to_m[ordering], unique_l_f, unique_m, unique_n)

def plot(path, distance_estimates, actual_penalties, l_fraction_values, m_values, n_values):
    stuff_to_unpack = rearrange_data_to_make_it_plottable(path, distance_estimates, actual_penalties, l_fraction_values, m_values, n_values)
    (distance_estimates_plot, relative_errors_plot, penalties_plot, relative_errors_ratios_plot, ratio_n_to_m, unique_l_f, unique_m, unique_n) = stuff_to_unpack



    fig, ax = plt.subplots(len(unique_m), 2, figsize=(16, 16))

    x_label_list = [f'{l:.1f}' for l in unique_l_f]


    min_penalty = 0
    max_penalty = 0.1
    min_err = -0.2
    max_err = 0.2

    plt.rc('axes', labelsize=16) #fontsize of the x and y labels

    img = []

    print('Plotting data ...')

    for k in range(len(unique_m)):
        img.append(ax[k,0].imshow(relative_errors_plot[k,:,:,plot_uncertainties_par], vmin=min_err, vmax=max_err))
        img.append(ax[k,1].imshow(penalties_plot[k,:,:,plot_uncertainties_par], vmin=min_penalty, vmax=max_penalty))

        #choose title adaptively, according to whether they want to plot the relative errors themselves or their uncertainties
        title1 = {False: r'Relative error $\hat{\rho}_F/\rho_F-1$' + ' for $m={m}$'.format(m=unique_m[k]), 
                    True: r'Uncertainty $\Delta(\hat{\rho}_F/\rho_F-1)$' + ' for $m={m}$'.format(m=unique_m[k])}
        title2 = {False: r'Bound penalties $\mathcal{L}_b / \lambda$' + ' for $m={m}$'.format(m=unique_m[k]),
                    True: r'Uncertainty $\Delta(\mathcal{L}_b / \lambda)$' + ' for $m={m}$'.format(m=unique_m[k])}
        ax[k,0].set_title(title1[plot_uncertainties])
        ax[k,1].set_title(title2[plot_uncertainties])

        for j in range(2): #set axis labels
            #ax[k,j].set_xticks(np.arange(len(unique_n)) + 0.5)
            ax[k,j].set_yticks(np.arange(len(unique_n)) + 0.0)
            ax[k,j].set_yticklabels(unique_n)
            ax[k,j].set_ylabel(r'$n$')
            ax[k,j].set_xlabel(r'$l/n$')
            ax[k,j].set_xticklabels(x_label_list)#[::5] + x_label_list[-2:-1])
            ax[k,j].set_xticks(np.arange(len(x_label_list)))
    
            divider = make_axes_locatable(ax[k,j])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(img[2*k + j], cax=cax, orientation='vertical')

    fig2, ax2 = plt.subplots(1,1)
    k = 5 #how many ticks to skip on each axis. k=5 looks nice
    plt.rc('axes', labelsize=30) #fontsize of the x and y labels
    img2 = ax2.imshow(relative_errors_ratios_plot, vmin=min_err, vmax=max_err)
    x_label_list = unique_l_f[::k]
    y_label_list = ratio_n_to_m[::k]
    ax2.set_xticks(np.arange(len(unique_l_f))[::k])
    ax2.set_yticks(np.arange(len(ratio_n_to_m))[::k])
    ax2.set_xticklabels(x_label_list)
    ax2.set_yticklabels(y_label_list)
    title3 = {False: r'$\hat{\rho}_F/\rho-1$', True:r'$\Delta(\hat{\rho}_F/\rho_F-1)$'}
    ax2.set(xlabel=r'$l/n$', ylabel=r'mass ratio $n/m$', title=title3[plot_uncertainties])
    fig2.colorbar(img2)

    """
    legacy plots: histograms of relative errors. See how they scatter

    ax[2,0].hist(relative_errors[0,:,:,0], bins=np.arange(0.8, 2.2, 0.1))
    ax[2,1].hist(relative_errors[1,:,:,0], bins=np.arange(0.8, 2.2, 0.1))
    """
    
    path_to_save = asksaveasfile(mode='w', defaultextension=".png")
    if path_to_save is not None: # asksaveasfile return `None` if dialog closed with "cancel".
        tmp = {False: '', True: '_uncertainties'}[plot_uncertainties]
        #truncate by name[:-4] to remove '.png' provided by user (we add it later on). Do this such that we can squeeze in a '_uncertainties' if need to in the filename
        fig.savefig(path_to_save.name[:-4] + tmp + '.png', format='PNG', dpi=300, bbox_inches = "tight")
        fig2.savefig(path_to_save.name[:-4] + tmp + '_ratio.png', format='PNG', dpi=300, bbox_inches = "tight")
    path_to_save.close()
    #os.remove(path_to_save.name) #remove file stub (asksaveasfile will create empty file; will not be used as file name is altered by the tmp string)


def main():
    path, dim = init()
    distance_estimates, actual_penalties, l_fraction_values, m_values, n_values = load_data(path, dim)
    plot(path, distance_estimates, actual_penalties, l_fraction_values, m_values, n_values)

#Execute worklfow
if __name__ == "__main__":
    main()
