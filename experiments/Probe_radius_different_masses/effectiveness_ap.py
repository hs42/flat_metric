import numpy as np 
import numpy.ma as ma
from matplotlib import pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tkinter as tk
from tkinter.filedialog import askdirectory, asksaveasfile
import os
import json

#important variables
data_points_to_consider = 50


def load_data(path_with, path_without, dim_by_user):
    for k, path in enumerate([path_with, path_without]):

        dirs = os.listdir(path)
        number_of_dirs = int(len(dirs))

        r_values = np.zeros(number_of_dirs)
        m_values = np.zeros(number_of_dirs, dtype=int)
        n_values = np.zeros(number_of_dirs, dtype=int)


        distance_estimates = np.zeros((number_of_dirs, 2, 2)) #dir_idx,[mean, std]
        relative_errors = np.zeros((number_of_dirs, 2, 2))    #dir_idx,[mean, std]
        actual_penalties = np.zeros((number_of_dirs, 2, 2))   #dir_idx,[mean, std]

        print('Loading data {x} adaptive penalty...'.format(x=['with', 'without'][k]))

        for i, dir in enumerate(dirs):
            with open(os.path.join(path, dir, 'logs{s}config.json'.format(s=os.sep))) as f:
                data = json.load(f)

            #r = abs(data['distrib2']['radius'] - data['distrib1']['radius'])
            r = data['distrib2']['radius']
            n = data['distrib2']['sample_size']
            m = data['distrib1']['sample_size']

            dim = data['distrib2']['dim']
            if dim != dim_by_user:
                #print(dim, dim_by_user)
                raise RuntimeError('The dimensions found do not match with the dimension label of the directory')
            linear_type = data['model']['linear']['type']

            #read in data
            distance_train = np.loadtxt(os.path.join(path, dir, 'logs{s}train_loss_W.log'.format(s=os.sep)), skiprows=1, delimiter=',')
            loss_flat = np.loadtxt(os.path.join(path, dir, 'logs{s}train_loss_flat.log'.format(s=os.sep)), skiprows=1, delimiter=',')
            lambda_penalties = np.loadtxt(os.path.join(path, dir, 'logs{s}lambda.log'.format(s=os.sep)), skiprows=1)


            #store important data
            distance_estimates[i, 0, k] = -np.mean(distance_train[:-data_points_to_consider:-1,1]) #mean of last data_points_to_consider entries
            distance_estimates[i ,1, k] = np.std(distance_train[:-data_points_to_consider:-1,1]) #std of last data_points_to_consider entries

            actual_penalties[i, 0, k] = np.mean(lambda_penalties[:-data_points_to_consider:-1,1]) #mean of last 10 entries of actual_penalty
            actual_penalties[i, 1, k] = np.std(lambda_penalties[:-data_points_to_consider:-1,1]) #mean of last 10 entries of actual_penalty
            
            groundtruth = min(2.0, r) + np.abs(n-m)/min(n, m)

            relative_errors[i, 0, k] = distance_estimates[i, 0, k] / groundtruth  - 1 
            relative_errors[i, 1, k] = distance_estimates[i, 1, k] / groundtruth
            

            r_values[i] = r
            m_values[i] = int(m)
            n_values[i] = int(n)

    return relative_errors


def plot(relative_errors):
    fig, ax = plt.subplots(1, 2, figsize=(16, 16))
    ax[0].hist(relative_errors[:,0,0], bins=20)
    ax[0].set_title('With adaptive penalty')
    ax[1].hist(relative_errors[:,0,1], bins=20)
    ax[1].set_title('Without adaptive penalty')
    fig.savefig