"""
Script for visualizing how flat distance estimates change over radius. Will produce plots like the subfigures 14A - 14E.
Essentially, the same script as Probe_radius_same_mass/visualize_estimates_over_radii.py only that the ground_truth was adjusted
"""

import numpy as np 
from matplotlib import pyplot as plt
import os
import json
import tkinter as tk
from tkinter.filedialog import askdirectory

dim = 0


data_points_to_consider = 50

root = tk.Tk()
root.withdraw()
path = askdirectory(title='Select folder containing the out files of a \'case_study\'-experiment')

if not os.path.exists(path):
    raise RuntimeError('Error provided path does not exist')

dirs = os.listdir(path)
r = np.zeros(len(dirs))
flat_metric = np.zeros((len(dirs),2))



for i, dir in enumerate(dirs):
    with open(os.path.join(path, dir, 'logs{s}config.json'.format(s=os.sep))) as f:
        data = json.load(f)
    r[i] = abs(data['distrib2']['radius'] - data['distrib1']['radius'])
    dim = data['distrib2']['dim']
    linear_type = data['model']['linear']['type']
    distance_train = np.loadtxt(os.path.join(path, dir, 'logs{s}train_loss_W.log'.format(s=os.sep)), skiprows=1, delimiter=',')
    flat_metric[i,0] = -np.mean(distance_train[:-data_points_to_consider:-1,1]) #mean of last data_points_to_consider entries
    flat_metric[i,1] = np.std(distance_train[:-data_points_to_consider:-1,1]) / np.sqrt(data_points_to_consider) #error of the mean
n = data['distrib2']['sample_size']
m = data['distrib1']['sample_size']

truth_x = np.linspace(min(r), max(r), 1000)
truth_y = groundtruth = np.minimum(2.0*np.ones_like(truth_x), truth_x) + np.abs(n-m)/min(n, m)

#plotting
plt.rc('font', size=20) #controls default text sizeplt.errorbar(r, flat_metric[:,0], yerr=flat_metric[:,1], marker=".", color='r',ms=20, label='experiment')


plt.errorbar(r, flat_metric[:,0], yerr=flat_metric[:,1], fmt=".", color='r',ms=13, 
    label='Experiment', elinewidth=2)
plt.plot(truth_x, truth_y, label='Ground truth', linewidth=3)
plt.ylim([3.5,7.5])
plt.legend()
plt.xlabel(r'Radius $r_0$ of distribution $\nu$')
plt.ylabel(r'$\rho_F(\mu, \nu)$')
plt.savefig('vary_r_dim={a}.png'.format(a=dim), format='PNG',dpi=300, bbox_inches = "tight")