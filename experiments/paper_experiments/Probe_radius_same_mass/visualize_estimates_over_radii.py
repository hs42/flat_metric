"""
This script will create plots akin to figure 1. Plot the flat distance estimates over the probed radii
"""

import numpy as np 
from matplotlib import pyplot as plt
import os
import json
import tkinter as tk
from tkinter.filedialog import askdirectory

__basedir__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir)
__filedir__ = os.path.dirname(os.path.abspath(__file__))

dim = 0

data_points_to_consider = 50 #number of last training epochs over which to take the average over


#ask user where the training files are stored
root = tk.Tk()
root.withdraw()
path = askdirectory(title='Select training folder containing the out files of a \'Probe radius with same masses\'-experiment', initialdir=os.path.join(__basedir__, 'out'))

if not os.path.exists(path):
    raise RuntimeError('Error provided path does not exist')

dirs = os.listdir(path)
r = np.zeros(len(dirs)) #create empty np array to store all the r values
flat_metric = np.zeros((len(dirs),2)) #create empty np array to store all the distance estimates as well as their uncertainties


#iteratively read in directories, i.e. loop over experiments which are trained with slightly different radii
for i, dir in enumerate(dirs):
    with open(os.path.join(path, dir, 'logs{s}config.json'.format(s=os.sep))) as f:
        data = json.load(f)
    r[i] = abs(data['distrib2']['radius'] - data['distrib1']['radius'])
    dim = data['distrib2']['dim']
    linear_type = data['model']['linear']['type']
    distance_train = np.loadtxt(os.path.join(path, dir, 'logs{s}train_loss_W.log'.format(s=os.sep)), skiprows=1, delimiter=',')
    flat_metric[i,0] = -np.mean(distance_train[:-data_points_to_consider:-1,1]) #mean of last data_points_to_consider entries
    flat_metric[i,1] = np.std(distance_train[:-data_points_to_consider:-1,1]) / np.sqrt(data_points_to_consider) #error of the mean

truth_x = np.linspace(min(r), max(r), 1000)
truth_y = [min(x,2.0) for x in truth_x] #in this simple experiment, the groundtruth is given as max(2, r_0)

#plotting
plt.rc('font', size=20) #controls default text sizeplt.errorbar(r, flat_metric[:,0], yerr=flat_metric[:,1], marker=".", color='r',ms=20, label='experiment')


plt.errorbar(r, flat_metric[:,0], yerr=flat_metric[:,1], fmt=".", color='r',ms=13, 
    label='Experiment', elinewidth=2)
plt.plot(truth_x, truth_y, label='Ground truth', linewidth=3)
plt.ylim([-0.1,2.3])
plt.legend()
plt.xlabel(r'Radius $r_0$ of distribution $\nu$')
plt.ylabel(r'$\rho_F(\mu, \nu)$')
#plt.title('Flat metric between two Diracs for dim={a} and {b} normalization'.format(a=dim, b=linear_type))
plt.savefig('vary_r_dim={a}.png'.format(a=dim), format='PNG',dpi=300, bbox_inches = "tight")