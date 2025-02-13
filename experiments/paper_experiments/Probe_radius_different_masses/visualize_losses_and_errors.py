"""
This script will create plots akin to figure 3 and figure 4. 
We make one plot with 6 subplots. In each of those wither the relative errors or the bound penalties are plotted as a function of the radius
and n, the mass of nu. 

In a separate plot, we collapse these plots and depict the relative error as a function of radius and mass ratio m/n
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



plot_uncertainties_par = {False: 0, True: 1}[plot_uncertainties] #convert boolean into an index which is specifically 0 or 1 -> can use that index in np arrays

def init():
    root = tk.Tk()
    root.withdraw()
    path = askdirectory(title='Select parent folder containing the out files of a \'Probe radius with different masses\'-experiment, \
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
    number_of_dirs = int(len(dirs))

    r_values = np.zeros(number_of_dirs)
    m_values = np.zeros(number_of_dirs, dtype=int)
    n_values = np.zeros(number_of_dirs, dtype=int)


    distance_estimates = np.zeros((number_of_dirs, 2)) #dir_idx,[mean, std]
    relative_errors = np.zeros((number_of_dirs, 2))    #dir_idx,[mean, std]
    actual_penalties = np.zeros((number_of_dirs, 2))   #dir_idx,[mean, std]

    print('Loading data ...')

    for i, dir in enumerate(dirs):
        with open(os.path.join(path, dir, 'logs{s}config.json'.format(s=os.sep))) as f:
            data = json.load(f)

        #r = abs(data['distrib2']['radius'] - data['distrib1']['radius'])
        r = data['distrib2']['radius']
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
        distance_estimates[i ,1] = np.std(distance_train[:-data_points_to_consider:-1,1])  / np.sqrt(data_points_to_consider)#error of the mean of last data_points_to_consider entries

        actual_penalties[i, 0] = np.mean(lambda_penalties[:-data_points_to_consider:-1,1]) #mean of last 10 entries of actual_penalty
        actual_penalties[i, 1] = np.std(lambda_penalties[:-data_points_to_consider:-1,1])  / np.sqrt(data_points_to_consider) #error of the mean of last data_points_to_consider entries

        
        groundtruth = min(2.0, r) + np.abs(n-m)/min(n, m)

        relative_errors[i, 0] = distance_estimates[i, 0] / groundtruth  - 1 
        relative_errors[i, 1] = distance_estimates[i, 1] / groundtruth
        

        r_values[i] = r
        m_values[i] = int(m)
        n_values[i] = int(n)

    return (distance_estimates, relative_errors, actual_penalties, r_values, m_values, n_values)

def rearrange_data_to_make_it_plottable(distance_estimates, relative_errors, actual_penalties, r_values, m_values, n_values):
    unique_m = np.unique(m_values)
    unique_n = np.unique(n_values)
    unique_r = np.unique(r_values)

    distance_estimates_plot = np.zeros((len(unique_m), len(unique_n), len(unique_r), 2))
    penalties_plot = np.zeros((len(unique_m), len(unique_n), len(unique_r), 2))
    relative_errors_plot = np.zeros((len(unique_m), len(unique_n), len(unique_r), 2))

    relative_errors_ratios_plot = np.zeros((len(unique_m) * len(unique_n), len(unique_r)))
    ratio_n_to_m = np.zeros(len(unique_m) * len(unique_n))

    print('Rearranging data ...')

    #most straightforward to iterate over parameter space
    for i, m in enumerate(unique_m):
        for j, n in enumerate(unique_n):
            for k, r in enumerate(unique_r):
                mask = np.logical_and(np.logical_and(m_values == m, n_values == n), r_values == r)

                distance_estimates_plot[i, j, k, :] = distance_estimates[mask, :]
                penalties_plot[i, j, k, :] = actual_penalties[mask, :]
                relative_errors_plot[i, j, k, :] = relative_errors[mask, :]
                relative_errors_ratios_plot[i*len(unique_n)+j, k] = relative_errors[mask, plot_uncertainties_par]
                ratio_n_to_m[i*len(unique_n)+j] = n/m

    ordering = np.argsort(ratio_n_to_m)

    return (distance_estimates_plot, relative_errors_plot, penalties_plot, relative_errors_ratios_plot[ordering, :], ratio_n_to_m[ordering], unique_r, unique_m, unique_n)

def plot(distance_estimates, relative_errors, actual_penalties, r_values, m_values, n_values):
    stuff_to_unpack = rearrange_data_to_make_it_plottable(distance_estimates, relative_errors, actual_penalties, r_values, m_values, n_values)
    (distance_estimates_plot, relative_errors_plot, penalties_plot, relative_errors_ratios_plot, ratio_n_to_m, unique_r, unique_m, unique_n) = stuff_to_unpack



    fig, ax = plt.subplots(len(unique_m), 2, figsize=(16, 16))



    min_penalty = 0
    max_penalty = 0.1
    min_err = -0.2
    max_err =  0.2

    if plot_uncertainties:
        min_penalty = 0
        max_penalty = 0.01
        min_err =  0.0
        max_err =  0.01

    r_start = 1 #index from which to start plotting radius

    x_label_list = [f'{r0:.1f}' for r0 in unique_r]
    x_label_list = x_label_list[r_start:]


    plt.rc('axes', labelsize=18) #fontsize of the x and y labels

    img = []

    print('Plotting data ...')
    titlesize = 18
    for k in range(len(unique_m)):
        img.append(ax[k,0].imshow(relative_errors_plot[k,:,r_start:,plot_uncertainties_par], vmin=min_err, vmax=max_err))#, vmin=0.0, vmax=2.0))# extent=[0,n_ax,0,n_ax], vmin=0, vmax=2,  cmap='hot'))
        img.append(ax[k,1].imshow(penalties_plot[k,:,r_start:,plot_uncertainties_par], vmin=min_penalty, vmax=max_penalty))#, extent=[0,n_ax,0,n_ax], vmin=min_penalty, vmax=max_penalty,  cmap='hot'))

        #choose title adaptively, according to whether they want to plot the relative errors themselves or their uncertainties
        title1 = {False: r'Relative error $\hat{\rho}_F/\rho_F-1$' + ' for $m={m}$'.format(m=unique_m[k]), 
                    True: r'Uncertainty $\Delta(\hat{\rho}_F/\rho_F-1)$' + ' for $m={m}$'.format(m=unique_m[k])}
        title2 = {False: r'Bound penalties $\mathcal{L}_b$' + ' for $m={m}$'.format(m=unique_m[k]),
                    True: r'Uncertainty $\Delta\mathcal{L}_b$' + ' for $m={m}$'.format(m=unique_m[k])}
        ax[k,0].set_title(title1[plot_uncertainties], fontsize=titlesize)
        ax[k,1].set_title(title2[plot_uncertainties], fontsize=titlesize)

        for j in range(2): #set axis labels
            #ax[k,j].set_xticks(np.arange(len(unique_n)) + 0.5)
            ax[k,j].set_yticks(np.arange(len(unique_n)) + 0.0)
            ax[k,j].set_yticklabels(unique_n)
            ax[k,j].set_ylabel(r'$n$')
            ax[k,j].set_xlabel(r'$r_0$', fontsize=18)
            ax[k,j].set_xticklabels(x_label_list)#[::5] + x_label_list[-2:-1])
            ax[k,j].set_xticks(np.arange(len(x_label_list)))
    
            divider = make_axes_locatable(ax[k,j])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(img[2*k + j], cax=cax, orientation='vertical')

    fig2, ax2 = plt.subplots(1,1)
    k = 5 #how many ticks to skip on each axis. k=5 looks nice
    plt.rc('axes', labelsize=30) #fontsize of the x and y labels
    img2 = ax2.imshow(relative_errors_ratios_plot[:, r_start:], vmin=min_err, vmax=max_err)
    x_label_list2 = x_label_list[::k]
    y_label_list = ratio_n_to_m[::k]
    ax2.set_xticks(np.arange(len(x_label_list))[::k])
    ax2.set_yticks(np.arange(len(ratio_n_to_m))[::k])
    ax2.set_xticklabels(x_label_list2)
    ax2.set_yticklabels(y_label_list)
    title3 = {False: r'$\hat{\rho}_F/\rho_F-1$', True:r'$\Delta(\hat{\rho}_F/\rho_F-1)$'}
    ax2.set(xlabel=r'$r_0$', ylabel=r'mass ratio $n/m$', title=title3[plot_uncertainties])
    fig2.colorbar(img2)

    
    path_to_save = asksaveasfile(mode='w', defaultextension=".eps")
    if path_to_save is not None: # asksaveasfile return `None` if dialog closed with "cancel".
        tmp = {False: '', True: '_uncertainties'}[plot_uncertainties]

        #truncate by name[:-4] to remove '.png' provided by user (we add it later on). Do this such that we can squeeze in a '_uncertainties' if need to in the filename
        fig.savefig(path_to_save.name[:-4] + tmp + '.eps', format='EPS', bbox_inches = "tight")
        fig2.savefig(path_to_save.name[:-4] + tmp + '_ratio.eps', format='EPS', bbox_inches = "tight")
    path_to_save.close()
    

def main():
    path, dim = init()
    distance_estimates, relative_errors, actual_penalties, r_values, m_values, n_values = load_data(path, dim)
    plot(distance_estimates, relative_errors, actual_penalties, r_values, m_values, n_values)

#Execute worklfow
if __name__ == "__main__":
    main()
