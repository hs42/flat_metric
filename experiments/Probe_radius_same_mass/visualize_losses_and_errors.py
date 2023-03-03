import numpy as np 
import numpy.ma as ma
from matplotlib import pyplot as plt
import os
import json
from tkinter.filedialog import askdirectory
from mpl_toolkits.axes_grid1 import make_axes_locatable

#path = askdirectory(title='Select folder containing the out files of a \'Probe radius with same masses\'-experiment where there are subfolders for many and few datapoints')
path='out/Probe_radius_same_mass/'

if not os.path.exists(path):
    raise RuntimeError('Error provided path does not exist')
if not os.path.exists(os.path.join(path, 'few_samples')):
    raise RuntimeError('Error: the few_samples subfolder does not exist')
if not os.path.exists(os.path.join(path, 'many_samples')):
    raise RuntimeError('Error: the many_samples subfolder does not exist')

tmp = os.listdir(os.path.join(path, 'many_samples'))
tmp.sort()

n_r = len(os.listdir(os.path.join(path, 'many_samples', tmp[0])))
#Assume that in each for each scenaria we conisder same dimensions and probed r values
n_dim = len(tmp)

r = np.zeros(n_r)
dims = []
data_p_to_consider = 50


flat_metric = {'many_samples': np.zeros((n_dim, n_r, 2)), 'few_samples' : np.zeros((n_dim, n_r, 2))}
loss_ratios = {'many_samples': np.zeros((n_dim, n_r, 2)), 'few_samples' : np.zeros((n_dim, n_r, 2))}
actual_penalties = {'many_samples': np.zeros((n_dim, n_r, 2)), 'few_samples' : np.zeros((n_dim, n_r, 2))}


for k, dim_dir in enumerate(tmp):
    for key in flat_metric:
        working_path = os.path.join(path, key, dim_dir)
        dirs = os.listdir(working_path)
        dirs.sort() #assuming that probed r values are same in each plot, this makes it possible to just store r values once without messing up the read-in data

        for i, dir in enumerate(dirs):
            with open(os.path.join(working_path, dir, 'logs{s}config.json'.format(s=os.sep))) as f:
                data = json.load(f)
            r[i] = abs(data['distrib2']['radius'] - data['distrib1']['radius'])
            linear_type = data['model']['linear']['type']

            distance_train = np.loadtxt(os.path.join(working_path, dir, 'logs{s}train_loss_W.log'.format(s=os.sep)), skiprows=1, delimiter=',')
            loss_flat = np.loadtxt(os.path.join(working_path, dir, 'logs{s}train_loss_flat.log'.format(s=os.sep)), skiprows=1, delimiter=',')
            lambda_penalties = np.loadtxt(os.path.join(working_path, dir, 'logs{s}lambda.log'.format(s=os.sep)), skiprows=1)


            flat_metric[key][k, i,0] = -np.mean(distance_train[:-data_p_to_consider:-1,1]) #mean of last 10 entries
            flat_metric[key][k, i,1] = np.std(distance_train[:-data_p_to_consider:-1,1]) / np.sqrt(data_p_to_consider)# error of mean

            actual_penalties[key][k, i,0] = np.mean(lambda_penalties[:-data_p_to_consider:-1,1]) #mean of last 10 entries
            actual_penalties[key][k, i,1] = np.std(lambda_penalties[:-data_p_to_consider:-1,1]) / np.sqrt(data_p_to_consider)# error of mean

            tmp2 = np.mean(loss_flat[:-data_p_to_consider:-1,1]) 
            tmp2_std = np.std(loss_flat[:-data_p_to_consider:-1,1]) / np.sqrt(data_p_to_consider)# error of mean

            loss_ratios[key][k,i,0] = - tmp2 / flat_metric[key][k,i,0] #ratio of mean of last data_p_to_consider entries
            
            with np.errstate(divide='ignore'): #ignore 'divide by 0' and resulting NaNs for now
                loss_ratios[key][k,i,1] =  np.sqrt((flat_metric[key][k,i,1]/flat_metric[key][k,i,0])**2 + (tmp2_std/tmp2)**2) #error propagation part 1


        loss_ratios[key][:,:,1] = np.where(np.isfinite(loss_ratios[key][:,:,1]), loss_ratios[key][:,:,1], 0)
        loss_ratios[key][:,:,1] = loss_ratios[key][:,:,0] * loss_ratios[key][:,:,1] #error propagation part 2
    
    dims.append(int(data['distrib2']['dim']))


truth_y = [min(x,2.0) for x in r]

relative_error = np.zeros((len(flat_metric)*len(dims),len(r), 2))
actual_penalties_to_plot = np.zeros((len(flat_metric)*len(dims),len(r), 2))

y_label_list = np.empty(len(flat_metric)*len(dims), dtype=object)

r_index_to_befin_plotting = 2

x_label_list = [f'{r0:.1f}' for r0 in r[r_index_to_befin_plotting:]]


plt.rc('axes', labelsize=13) #fontsize of the x and y labels
fig, axs = plt.subplots(2,2, figsize=(15,5), sharex=True)
fig.tight_layout(pad=5.0)

# bring data in format to be plotted
for k, dim in enumerate(dims):
    for l, key in enumerate(flat_metric):
        relative_error[k+l*len(dims),r_index_to_befin_plotting:, 0] = - 1 + flat_metric[key][k, r_index_to_befin_plotting:,0] / truth_y[r_index_to_befin_plotting:]
        relative_error[k+l*len(dims),r_index_to_befin_plotting:, 1] = flat_metric[key][k, r_index_to_befin_plotting:,1] / flat_metric[key][k, r_index_to_befin_plotting:,0]

        actual_penalties_to_plot[k+l*len(dims),r_index_to_befin_plotting:,:] = actual_penalties[key][k,r_index_to_befin_plotting:,:]

        y_label_list[k+l*len(dims)] = 'd=' + str(dim) + ', ' + key[0]

# plot 4 subplots
minerr = [-0.2, 0.0]
maxerr = [0.2, 0.01]

minbound = [0.0, 0.0]
maxbound = [0.1, 0.01]

for i in range(2):
    for j in range(2):
        if i == 0:
            im = axs[i,j].imshow(relative_error[:,r_index_to_befin_plotting:,j], vmin=minerr[j], vmax=maxerr[j])

        else:
            im = axs[i,j].imshow(actual_penalties_to_plot[:,r_index_to_befin_plotting:,j], vmin=minbound[j], vmax=maxbound[j])
        axs[i,j].set_xticklabels(x_label_list[::5])
        axs[i,j].set_xticks(np.arange(len(x_label_list))[::5])
        axs[i,j].set_xlabel(r'$r_0$')
        axs[i,j].set_yticklabels(y_label_list)
        axs[i,j].set_yticks(np.arange(len(y_label_list)))
        divider = make_axes_locatable(axs[i,j])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

  
axs[0,1].set_title(r'Uncertainties of these relative errors $\Delta(\hat{\rho}_F/\rho_F-1)$')

axs[0,0].set_title(r'Relative error $\hat{\rho}_F/\rho_F-1$')
axs[1,0].set_title(r'Bound penalties $\mathcal{L}_b$')

axs[1,1].set_title(r'Uncertainties of these penalties $\Delta \mathcal{L}_b$')


fig.savefig('estimates_over_ground_truth.png', dpi=300, format='PNG',bbox_inches = "tight")