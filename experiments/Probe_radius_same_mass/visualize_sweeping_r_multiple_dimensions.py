import numpy as np 
from matplotlib import pyplot as plt
import os
import json
import tkinter as tk
from tkinter.filedialog import askdirectory


#path = 'out{s}examine_loss_and_errors{s}just_a_simple_test'.format(s=os.sep)


data_p_to_consider = 50

root = tk.Tk()
root.withdraw()
#path='out/Probe_radius_same_mass/'
path = askdirectory(title='Select folder containing directories of \'Probe radius with same masses\'-experiments over several dimensions')
#path = input("Enter path to the out files of a \'Probe radius with same masses\'-experiment:")

if not os.path.exists(path):
    raise RuntimeError('Error provided path does not exist')

subdirs = os.listdir(path)
subdirs.sort()
dirs = os.listdir(os.path.join(path, subdirs[0]))
r = np.zeros((len(subdirs), len(dirs))) 
flat_metric = np.zeros((len(subdirs), len(dirs), 2))
dims = np.zeros(len(subdirs))


for j, experiment in enumerate(subdirs):
    dirs = os.listdir(os.path.join(path, experiment))


    for i, dir in enumerate(dirs):
        with open(os.path.join(path, experiment, dir, 'logs{s}config.json'.format(s=os.sep))) as f:
            data = json.load(f)
        r[j, i] = abs(data['distrib2']['radius'] - data['distrib1']['radius'])
        dims[j] = data['distrib2']['dim']
        linear_type = data['model']['linear']['type']
        distance_train = np.loadtxt(os.path.join(path, experiment, dir, 'logs{s}train_loss_W.log'.format(s=os.sep)), skiprows=1, delimiter=',')
        flat_metric[j, i,0] = -np.mean(distance_train[:-data_p_to_consider:-1,1]) #mean of last data_p_to_consider entries
        flat_metric[j, i,1] = np.std(distance_train[:-data_p_to_consider:-1,1]) / np.sqrt(data_p_to_consider) #error of the mean

truth_x = np.linspace(np.amin(r), np.amax(r))
truth_y = [min(x,2.0) for x in truth_x]

plt.rc('font', size=20) #controls default text sizeplt.errorbar(r, flat_metric[:,0], yerr=flat_metric[:,1], marker=".", color='r',ms=20, label='experiment')

colors_for_scatter = ['r', 'g', 'b', 'y']
labels_for_scatter = ['.', 's', 'd', 'p']

for j, experiment in enumerate(subdirs):
    plt.errorbar(r[j,:], flat_metric[j, :,0], yerr=3*flat_metric[j, :,1], marker=labels_for_scatter[j], linestyle='none', color=colors_for_scatter[j],ms=8, 
        label=r'experiment, $d={d}$'.format(d=int(dims[j])), elinewidth=2)
plt.plot(truth_x, truth_y, label='ground truth', linewidth=3)
plt.ylim([-0.1,2.5])
plt.xlim([0,2.5])
plt.legend()
plt.xlabel(r'radius $r_0$ of $\nu=N\delta(r-r_0)$')
plt.ylabel(r'$W_1^{1,1}(\mu, \nu)$')
#plt.title('Flat metric between two Diracs for dim={a} and {b} normalization'.format(a=dim, b=linear_type))
plt.savefig('vary_r.png', format='PNG',dpi=300, bbox_inches = "tight")